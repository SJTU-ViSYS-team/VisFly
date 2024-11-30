from typing import Callable, Union


def linear_schedule(initial: float, final: float = 0.) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * (initial-final) + final

    return func


def exponential_schedule(initial: float, decay: float) -> Callable[[float], float]:
    """
    Exponential learning rate schedule.

    :param initial: Initial learning rate.
    :param decay: Rate of decay.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return initial * (decay ** (1 - progress_remaining))

    return func


def transfer_schedule(learning_rate: Union[dict, float]) -> Callable[[float], float]:
    """
    Transfer learning rate schedule.

    :param kwargs: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    class_alias = {
        'linear': linear_schedule,
        'exponential': exponential_schedule
    }
    if isinstance(learning_rate, dict):
        schedule_class = class_alias[learning_rate['class']]
        return schedule_class(**learning_rate['kwargs'])
    elif isinstance(learning_rate, (int, float)):
        return learning_rate
    else:
        raise ValueError(f"Invalid learning rate type: {type(learning_rate)}")
