from torch import nn


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class NLinear(nn.Sequential):
    def __init__(self, in_feat, units, act=nn.ELU):
        layers = [nn.Linear(in_feat, units[0])]
        for i in range(len(units) - 1):
            in_feat, out_feat = units[i:i + 2]
            layers.append(act())
            layers.append(nn.Linear(in_feat, out_feat))

        super(NLinear, self).__init__(*layers)
