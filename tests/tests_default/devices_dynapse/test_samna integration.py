""" 
The current dynap-se2 API support avoids using samna objects directly in dynap-se2 application software support pipeline. 
Instead it copies the data-segment from the samna objects to samna_alias objects whenever needed

The workflow is samna.to_json -> alias.from_json -> play and manipulate inside rockpool -> alias.to_json -> samna.from_json

This test package checks if the conversion is healthy and successful

In case of a failure, check
    * samna
    * .rockpool/devices/dynapse/samna_alias/se2

Notes:
    * ``test_full_config`` failure is less crucial then test_event_conversion
        * ``test_full_config`` fails : one cannot save and reload samna configuration objects reliably
        * ``test_event`` or ``test_destination_key`` fails : DynapSamna interfacing, especially AER package delivery does not work
"""

import pytest


def test_full_config():
    """
    test_full_config looks small but it checks the full samna SE2 alias conversion pipeline in one shot
    It test passes only if whole 200MB configuration package can correctly be serialized, and deserialized.

    The most probable reasons for failure are
    * updates in samna
        * snake_cakse to camelCase and camelCase to snake_case conversion fixes
            * now buggy 221202
        * field renaming
    """
    import samna
    from rockpool.devices.dynapse import samna_alias

    original = samna.dynapse2.Dynapse2Configuration()
    alias = samna_alias.Dynapse2Configuration.from_samna(original)
    assert original.to_json() == alias.to_json()


def test_event():
    """
    test_event checks if the AER package conversion works succesfully
    It passes only if AER packages can be resolved succesfully

    The most probable reasons for failure are
    * custom hash implementation for Dynapse2Destination object
    * updates in samna
    """
    import samna
    from rockpool.devices.dynapse import samna_alias

    # Randomly selected values
    core = [True, False, True, False]
    x_hop = -1
    y_hop = 1
    tag = 1234
    event_time = 12.56
    dt_fpga = 1e-6

    alias_event = samna_alias.NormalGridEvent(
        event=samna_alias.Dynapse2Destination(core, x_hop, y_hop, tag),
        timestamp=int(event_time / dt_fpga),
    )
    samna_event: samna.dynapse2.NormalGridEvent = alias_event.to_samna()

    assert samna_event.to_json() == alias_event.to_json()


def test_destination_key():
    """
    test_event_sorting checks if ``Dynapse2Destinations`` can be used as keys succesfully
    It passes only if the custom hash implementation for ``Dynapse2Destination`` is functioning properly

    The most probable reasons for failure are
    * ...samna_alias/se2/Dynapse2Destination.__hash__
    """
    from rockpool.devices.dynapse import samna_alias

    # Randomly selected values
    core_list = [
        [True, False, True, False],
        [False, False, True, False],
        [True, False, True, True],
        [True, True, True, False],
        [True, True, True, True],
        [False, False, False, False],
        [False, False, True, False],
    ]

    x_hop_list = [-1, 0, 3, 4, -5, 6, 7]
    y_hop_list = [0, 1, 2, -3, 4, 5, 6]
    tag_list = [10, 20, 2000, 1000, 48, 1024, 32]
    times = [1000, 1023, 1056, 1068, 1096, 2022, 2023]

    # Create the sequence
    seq = []
    for core, x_hop, y_hop, tag in zip(core_list, x_hop_list, y_hop_list, tag_list):
        seq.append(samna_alias.Dynapse2Destination(core, x_hop, y_hop, tag))

    __dict = dict(zip(seq, times))

    # Construct a new object
    key4 = samna_alias.Dynapse2Destination(
        core=core_list[4], x_hop=x_hop_list[4], y_hop=y_hop_list[4], tag=tag_list[4]
    )

    assert __dict[key4] == times[4]
