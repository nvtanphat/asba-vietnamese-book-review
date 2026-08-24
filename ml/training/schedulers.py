from transformers import get_linear_schedule_with_warmup
def linear_warmup(optimizer,total_steps,warmup_ratio=0.1): return get_linear_schedule_with_warmup(optimizer,int(total_steps*warmup_ratio),total_steps)
