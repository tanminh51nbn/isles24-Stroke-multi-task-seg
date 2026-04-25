import torch.nn as nn

class MultiTaskHeads(nn.Module):
    """
    Independent 1x1 Convolutional heads for each task.
    """
    def __init__(self, in_channels: int = 16, tasks_cfg: dict = None):
        super().__init__()
        # Mặc định là 1 channel output cho mỗi task nếu không có config
        lesion_out = tasks_cfg.get("lesion", {}).get("out_channels", 1) if tasks_cfg else 1
        lvo_out    = tasks_cfg.get("lvo", {}).get("out_channels", 1) if tasks_cfg else 1
        cow_out    = tasks_cfg.get("cow", {}).get("out_channels", 1) if tasks_cfg else 1
        
        dropout_rate = tasks_cfg.get("dropout", 0.2) if tasks_cfg else 0.2
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        self.lesion = nn.Conv2d(in_channels, lesion_out, kernel_size=1)
        self.lvo = nn.Conv2d(in_channels, lvo_out, kernel_size=1)
        self.cow = nn.Conv2d(in_channels, cow_out, kernel_size=1)
        # Mỗi head là Conv1×1 độc lập — output raw logits, không sigmoid
    def forward(self, x):
        x = self.dropout(x)
        return self.lesion(x), self.lvo(x), self.cow(x)
