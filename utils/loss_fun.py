def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T) / temp
    brain_clip = (preds @ targs.T) / temp
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()

    loss = (loss1 + loss2) / 2
    return loss