""" Neural Network Model

"""


# region Imported Dependencies
import importlib
import torch
from torch import nn, optim
from brain.nn.util.param import Param
# endregion Imported Dependencies


class Net:
    def __init__(self, a_arch: Param, a_optim: Param, a_lrs: Param, a_loss: Param, a_device: torch.device,
                 a_half: bool) -> None:
        self.arch_param: Param = a_arch
        self.optim_param: Param = a_optim
        self.lrs_param: Param = a_lrs
        self.loss_param: Param = a_loss
        self.device: torch.device = a_device
        self.half: bool = a_half

        # Initialize Architecture
        self.arch: nn.Module = None
        self.__init_arch()

        # Initialize Optimizer
        self.optim: optim = None
        self.__init_optim()

        # Initialize Learning Rate Scheduler
        self.lrs: optim.lr_scheduler = None
        self.__init_lrs()

        # Initialize Loss Function
        self.loss: callable = None
        self.__init_loss()

    def __init_arch(self) -> None:
        try:
            # Build Architecture
            module_name = f"brain.nn.{self.arch_param.name}.arch"
            module = importlib.import_module(module_name)
            arch_class = getattr(module, 'Arch')
            self.arch: nn.Module = arch_class(**self.arch_param.kwargs)
        except (AttributeError, ModuleNotFoundError):
            raise ValueError('Invalid `arch.name` is entered.')

        # Target Device
        self.arch.to(self.device)

        # Half Precision
        if self.half:
            self.arch.half()

    def __init_optim(self) -> None:
        try:
            optimizer = getattr(optim, self.optim_param.name, None)
        except ModuleNotFoundError:
            raise ValueError(f'Invalid `optim.name` is entered: {self.optim_param.name}')

        self.optim = optimizer(params=self.arch.parameters(), **self.optim_param.kwargs)

    def __init_lrs(self) -> None:
        try:
            lrs = getattr(optim.lr_scheduler, self.lrs_param.name, None)
        except ModuleNotFoundError:
            raise ValueError(f'Invalid `lrs.name` is entered: {self.lrs_param.name}')

        self.lrs = lrs(optimizer=self.optim, **self.lrs_param.kwargs)

    def __init_loss(self) -> None:
        try:
            loss = getattr(nn, self.loss_param.name, None)
        except ModuleNotFoundError:
            raise ValueError(f'Invalid `loss.name` is entered: {self.loss_param.name}')

        self.loss = loss(**self.loss_param.kwargs)

    def train(self):
        NotImplemented

    def validate(self):
        # Save the best model
        NotImplemented

    def test(self):
        NotImplemented
