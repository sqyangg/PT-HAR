import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- 辅助函数 --------------------
def _pair(x):
    """将整数转换为二元组，保持元组不变"""
    return x if isinstance(x, tuple) else (x, x)

def get_act_layer(name):
    """简单的激活函数映射"""
    if name == 'swish':
        return nn.SiLU
    else:
        return nn.ReLU

# -------------------- 纯 PyTorch LocalConvolution --------------------
class LocalConvolution(nn.Module):
    """
    纯 PyTorch 实现的局部动态卷积，不依赖 cupy/CUDA。
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, pad_mode=0):
        super(LocalConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_mode = pad_mode

    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        kernel_size = _pair(self.kernel_size)
        stride = _pair(self.stride)
        padding = _pair(self.padding)
        dilation = _pair(self.dilation)

        B, C_in, H, W = input.shape
        B2, heads, C_w, kk, H_out, W_out = weight.shape
        assert B == B2, "batch size mismatch"
        assert C_in % C_w == 0, "C_in must be divisible by C_w"
        groups = C_in // C_w
        assert kk == kernel_size[0] * kernel_size[1], "weight's last spatial dim must equal prod(kernel_size)"

        # unfold提取滑动窗口
        x_unfold = F.unfold(
            input,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride
        )  # (B, C_in * kk, L)
        L = x_unfold.shape[-1]
        x_unfold = x_unfold.view(B, groups, C_w, kk, H_out, W_out)

        # 动态加权求和
        out = torch.einsum('b h c k H W, b g c k H W -> b h g c H W', weight, x_unfold)
        out = out.reshape(B, heads * groups * C_w, H_out, W_out)
        return out

# -------------------- 修改后的 CoT Layer（仅横向）--------------------
class PatLayer(nn.Module):
    """
    横向划分的 Contextual Transformer 模块，可替代 ResNet 中的 3×3 卷积。
    """
    def __init__(self, dim, kernel_size):
        super(PatLayer, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        # 静态上下文提取：(1, kernel_size) 卷积
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, self.kernel_size),
                      stride=1, padding=(0, self.kernel_size // 2),
                      groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        group_size = dim // share_planes  # 确保整除

        # 动态注意力权重生成：输出 kernel_size * group_size 个通道
        self.embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, self.kernel_size * group_size, kernel_size=1),
            nn.GroupNorm(num_groups=share_planes,
                         num_channels=self.kernel_size * group_size)
        )

        # 1×1 值变换
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        # 局部动态卷积（仅横向）
        self.local_conv = LocalConvolution(
            dim, dim,
            kernel_size=(1, self.kernel_size),
            stride=1,
            padding=(0, (self.kernel_size - 1) // 2),
            dilation=1
        )
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU(inplace=True)  # Swish

        # 融合注意力模块（SE style）
        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix * dim, 1)
        )

    def forward(self, x):
        # 静态上下文
        k = self.key_embed(x)                     # (B, dim, H, W)

        # 拼接查询与静态键
        qk = torch.cat([x, k], dim=1)              # (B, 2*dim, H, W)
        B, _, H, W = qk.shape

        # 生成动态注意力权重
        w = self.embed(qk)                         # (B, kernel_size * group_size, H, W)
        group_size = self.dim // 8                  # share_planes = 8
        w = w.view(B, 1, group_size, self.kernel_size, H, W)  # (B, heads=1, C_w, k, H, W)

        # 值变换
        v = self.conv1x1(x)                         # (B, dim, H, W)

        # 动态上下文聚合
        v = self.local_conv(v, w)                   # (B, dim, H, W)
        v = self.bn(v)
        v = self.act(v)

        # 融合静态和动态上下文
        v = v.view(B, self.dim, 1, H, W)
        k = k.view(B, self.dim, 1, H, W)
        fused = torch.cat([v, k], dim=2)            # (B, dim, 2, H, W)

        # 全局注意力加权
        fused_gap = fused.sum(dim=2)                 # (B, dim, H, W)
        fused_gap = fused_gap.mean((2, 3), keepdim=True)  # (B, dim, 1, 1)
        '''
        ########
        attn = self.se(fused_gap)                    # (B, radix*dim, 1, 1)
        attn = attn.view(B, self.dim, self.radix)    # (B, dim, radix)
        attn = F.softmax(attn, dim=2)

        out = (fused * attn.reshape(B, self.dim, self.radix, 1, 1)).sum(dim=2)
        #########
        '''
        out = fused.sum(dim=2)
        return out.contiguous()

# -------------------- 简化版 SE 模块 --------------------
class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

def create_attn(attn_layer, channels):
    """简单工厂，根据字符串创建注意力模块"""
    if attn_layer is None:
        return None
    if attn_layer == 'se':
        return SELayer(channels)
    # 可根据需要扩展其他注意力
    return None

# -------------------- Bottleneck 模块 --------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        # 可选的下采样前平均池化（抗锯齿）
        self.avd = nn.AvgPool2d(3, 2, padding=1) if stride > 1 else None

        # 将 3x3 卷积替换为 CoT 模块
        self.conv2 = PatLayer(width, kernel_size=3)

        # 原 conv2 后的 BN、激活和抗锯齿层均被移除，因为 CotLayer 内部已包含

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        if self.avd is not None:
            x = self.avd(x)

        x = self.conv2(x)          # CoT 模块

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.act3(x)

        return x

# -------------------- 简化版 ResNet --------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, dataset, in_chans=3, cardinality=1, base_width=64,
                 stem_width=64, stem_type='', replace_stem_pool=False, block_reduce_first=1,
                 down_kernel_size=1, avg_down=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 aa_layer=None, attn_layer=None, drop_rate=0.0, drop_path_rate=0.0,
                 drop_block_rate=0.0, zero_init_last_bn=True, **kwargs):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.cardinality = cardinality
        self.base_width = base_width
        self.aa_layer = aa_layer
        self.attn_layer = attn_layer
        self.down_kernel_size = down_kernel_size
        self.avg_down = avg_down
        self.replace_stem_pool = replace_stem_pool
        self.stem_width = stem_width   # 保存 stem_width
        self.dataset = dataset
        
        if self.dataset == 'UT_HAR_data':
            #################### for UT_HAR dataset#####
            self.reshape = nn.Sequential(
                nn.Conv2d(1,3,7,stride=(3,1)),
                nn.ReLU(),
                #nn.MaxPool2d(2),
                nn.Conv2d(3,3,kernel_size=(10,11),stride=1),
                nn.ReLU()
                )
            self.num_classes = 7
        else:
            #################### for UT_HAR dataset#####
                
            self.reshape = nn.Sequential(
                    nn.Conv2d(3,3,(15,23),stride=(3,9)),
                    nn.ReLU(),
                    nn.Conv2d(3,3,kernel_size=(3,23),stride=1),
                    nn.ReLU()
                    )
            self.num_classes = 6
        
        # Stem
        deep_stem = 'deep' in stem_type
        if deep_stem:
            self.conv1 = nn.Conv2d(in_chans, stem_width, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(stem_width)
            self.act1 = act_layer(inplace=True)
        else:
            self.conv1 = nn.Conv2d(in_chans, stem_width, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(stem_width)
            self.act1 = act_layer(inplace=True)

        # 池化
        if replace_stem_pool:
            self.maxpool = None
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inplanes = stem_width   # 当前特征图通道数

        # 构建四个 stage
        self.stage1 = self._make_stage(block, 64, layers[0], stride=1,
                                        reduce_first=block_reduce_first, attn_layer=attn_layer)
        self.stage2 = self._make_stage(block, 128, layers[1], stride=2,
                                        reduce_first=block_reduce_first, attn_layer=attn_layer)
        self.stage3 = self._make_stage(block, 256, layers[2], stride=2,
                                        reduce_first=block_reduce_first, attn_layer=attn_layer)
        self.stage4 = self._make_stage(block, 512, layers[3], stride=2,
                                        reduce_first=block_reduce_first, attn_layer=attn_layer)

        # 分类头
        num_features = 512 * block.expansion
        self.num_features = num_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def _make_stage(self, block, planes, blocks, stride=1, reduce_first=1, attn_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                    self.norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=self.down_kernel_size,
                              stride=stride, padding=self.down_kernel_size//2, bias=False),
                    self.norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.cardinality, self.base_width,
                            reduce_first, 1, None, self.act_layer, self.norm_layer, attn_layer, self.aa_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, self.cardinality, self.base_width,
                                reduce_first, 1, None, self.act_layer, self.norm_layer, attn_layer, self.aa_layer))

        return nn.Sequential(*layers)
    def forward(self, x):
        
        x = self.reshape(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.maxpool is not None:
            x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x

# -------------------- 模型创建函数 --------------------
def ptnet50(num_classes, dataset, pretrained=False, **kwargs):
    """构建 ptnet-50"""
    def __init__(self, num_classes, dataset):
       super(ptnet50, self).__init__()
       self.num_classes = num_classes
       self.dataset = dataset
    
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, dataset, stem_width=64, **kwargs)
    return model

def ptnet18(num_classes, dataset, pretrained=False, **kwargs):
    """构建 ptnet-18"""
    def __init__(self, num_classes, dataset):
       super(ptnet18, self).__init__()
       self.num_classes = num_classes
       self.dataset = dataset
    
    model = ResNet(Bottleneck, [2, 2, 2, 2], num_classes, dataset,stem_width=64, **kwargs)
    return model



def ptnet101(num_classes, dataset, pretrained=False, **kwargs):
    """构建 CoTNet-101"""
    def __init__(self, num_classes, dataset):
       super(ptnet101, self).__init__()
       self.num_classes = num_classes
       self.dataset = dataset
    
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, dataset, stem_width=64, **kwargs)
    return model



# -------------------- 测试 --------------------
if __name__ == '__main__':
    # 创建模型
    net = cotnet50(num_classes=6)
    #print(net)

    # 测试前向传播
    input_image = torch.randn(2, 3, 224, 224)
    output = net(input_image)
    print("Output shape:", output.shape)  # 应为 (2, 1000)