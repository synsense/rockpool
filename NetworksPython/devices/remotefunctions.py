# ----
# remotefunctions.py - Functions that are to be teleported and defined in the remote
#                      namespace if RPyC is used
# Author: Felix Bauer, aiCTX AG, felix.bauer@ai-ctx.com
# ----

__all__ = ["_auto_insert_dummies", "_replace_too_large_value", "local_arguments"]


def _auto_insert_dummies(
    discrete_isi_list: list, neuron_ids: list, fpga_isi_limit: int = FPGA_ISI_LIMIT
) -> (list, list):
    """
    auto_insert_dummies - Insert dummy events where ISI limit is exceeded
    :param discrete_isi_list:  list  Inter-spike intervals of events
    :param neuron_ids:     list  IDs of neurons corresponding to the ISIs

    :return:
        corrected_isi_list     list  ISIs that do not exceed limit, including dummies
        corrected_id_list      list  Neuron IDs corresponding to corrected ISIs. Dummy events have None.
        isdummy_list           list  Boolean indicating which events are dummy events
    """
    # - List of lists with corrected entries
    corrected: List[List] = [
        _replace_too_large_value(isi, fpga_isi_limit) for isi in discrete_isi_list
    ]
    # - Number of new entries for each old entry
    new_event_counts = [len(l) for l in corrected]
    # - List of lists with neuron IDs corresponding to ISIs. Dummy events have ID None
    id_lists: List[List] = [
        [id_neur, *(None for _ in range(length - 1))]
        for id_neur, length in zip(neuron_ids, new_event_counts)
    ]
    # - Flatten out lists
    corrected_isi_list = [isi for l in corrected for isi in l]
    corrected_id_list = [id_neur for l in id_lists for id_neur in l]
    # - Count number of added dummy events (each one has None as ID)
    num_dummies = len(tuple(filter(lambda x: x is None, corrected_id_list)))
    if num_dummies > 0:
        print("dynapse_control: Inserted {} dummy events.".format(num_dummies))

    return corrected_isi_list, corrected_id_list


def _replace_too_large_value(value, limit: int = FPGA_ISI_LIMIT):
    """
    replace_too_large_entry - Return a list of integers <= limit, that sum up to value
    :param value:    int  Value to be replaced
    :param limit:  int  Maximum allowed value
    :return:
        lnReplace   list  Values to replace value
    """
    if value > limit:
        reps = (value - 1) // limit
        # - Return reps times limit, then the remainder
        #   For modulus shift value to avoid replacing with 0 if value==limit
        return [*(limit for _ in range(reps)), (value - 1) % limit + 1]
    else:
        # - If clause in particular for case where value <= 0
        return [value]


def local_arguments(func):
    def local_func(*args, **kwargs):
        for i, argument in enumerate(args):
            newargs = list(args)
            if isinstance(argument, rpyc.core.netref.BaseNetref):
                newargs[i] = copy.copy(argument)
        for key, val in kwargs.items():
            if isinstance(key, rpyc.core.netref.BaseNetref):
                del kwargs[key]
                kwargs[copy.copy(key)] = copy.copy(val)
            elif isinstance(val, rpyc.core.netref.BaseNetref):
                kwargs[key] = copy.copy(val)

        return func(*newargs, **kwargs)

    return local_func


# # - Example on how to use local_arguments_rpyc decorator
# @teleport_function
# def _define_print_type():
#     @local_arguments
#     def print_type(obj):
#         print(type(obj))
#     return print_type
# print_type = correct_argument_types(
#     _define_print_type()
# )  # or just print_type = _define_print_type()
