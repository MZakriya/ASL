import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        batch, channel, t, h, w = x.size()
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)

class Unit3D(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1), stride=(1, 1, 1), padding=0, activation_fn=F.relu, use_batch_norm=True, use_bias=False, name='unit_3d'):
        super(Unit3D, self).__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=self._output_channels, kernel_size=self._kernel_shape, stride=self._stride, padding=0, bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()
        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0, name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0, name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3], name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0, name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3], name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0, name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)

class InceptionI3d(nn.Module):
    def __init__(self, num_classes=400, spatial_squeeze=True, final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if final_endpoint not in ['Logits', 'Mixed_5c']:
            raise ValueError('Unknown final_endpoint %s' % final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7], stride=[2, 2, 2], padding=3, name=name+end_point)
        self.end_points['MaxPool3d_2a_3x3'] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=0)
        self.end_points['Conv3d_2b_1x1'] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0, name=name+'Conv3d_2b_1x1')
        self.end_points['Conv3d_2c_3x3'] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1, name=name+'Conv3d_2c_3x3')
        self.end_points['MaxPool3d_3a_3x3'] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=0)
        self.end_points['Mixed_3b'] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name+'Mixed_3b')
        self.end_points['Mixed_3c'] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name+'Mixed_3c')
        self.end_points['MaxPool3d_4a_3x3'] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=0)
        self.end_points['Mixed_4b'] = InceptionModule(480, [192, 96, 208, 16, 48, 64], name+'Mixed_4b')
        self.end_points['Mixed_4c'] = InceptionModule(512, [160, 112, 224, 24, 64, 64], name+'Mixed_4c')
        self.end_points['Mixed_4d'] = InceptionModule(512, [128, 128, 256, 24, 64, 64], name+'Mixed_4d')
        self.end_points['Mixed_4e'] = InceptionModule(512, [112, 144, 288, 32, 64, 64], name+'Mixed_4e')
        self.end_points['Mixed_4f'] = InceptionModule(528, [256, 160, 320, 32, 128, 128], name+'Mixed_4f')
        self.end_points['MaxPool3d_5a_2x2'] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0)
        self.end_points['Mixed_5b'] = InceptionModule(832, [256, 160, 320, 32, 128, 128], name+'Mixed_5b')
        self.end_points['Mixed_5c'] = InceptionModule(832, [384, 192, 384, 48, 128, 128], name+'Mixed_5c')
        self.end_points['Logits'] = Unit3D(in_channels=1024, output_channels=num_classes, kernel_shape=[1, 1, 1], padding=0, activation_fn=None, use_batch_norm=False, use_bias=True, name='logits')
        self.end_points['Predictions'] = nn.Softmax(dim=1)

        self.build()

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        # x is [batch, channel, depth, height, width]
        for end_point in self.end_points:
            if end_point in ['Logits', 'Predictions']: continue
            x = self._modules[end_point](x) 
            if end_point == self._final_endpoint:
                break
        
        # If we stopped at Mixed_5c, we have [batch, 1024, d, h, w]
        if self._final_endpoint == 'Mixed_5c':
            # Average pool over spaial dimensions, keep temporal? As per original paper?
            # Or average pool everything?
            # The user code expects 1024 features per frame? Or per sequence?
            # User code: encoder_input shape (batch, seq, 2653).
            # This implies 2653 per FRAME.
            # But I3D is spatiotemporal (takes chunks of frames).
            # If we run I3D on a sliding window, we get features.
            # 1024 features from Mixed_5c are per-clip?
            # Usually we pool (AvgPool3d) to get [batch, 1024, 1, 1, 1] -> [batch, 1024].
            return x

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        logits = logits.mean(dim=2)
        return logits

    def extract_features(self, x):
        # Custom method to get 1024-dim features
        # x: [batch, 3, T, 224, 224]
        for end_point in self.end_points:
            if end_point == 'Logits': break
            x = self._modules[end_point](x)
        
        # x is [batch, 1024, T/8, H/32, W/32]
        # Avg pool over spatial (H, W)
        x = F.avg_pool3d(x, kernel_size=[1, x.size(3), x.size(4)], stride=[1, 1, 1])
        # x is [batch, 1024, T/8, 1, 1]
        x = x.squeeze(3).squeeze(3) # [batch, 1024, T/8]
        return x.transpose(1, 2) # [batch, T/8, 1024]
