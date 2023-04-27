# test lookup table

def test_lookup_table():
    from rockpool.devices.xylo.imu.preprocessing import RotationLookUpTable
    import numpy as np
    
    num_angles = 64
    num_bits = 16
    
    lut = RotationLookUpTable(
        num_angles=num_angles,
        num_bits=num_bits
    )
    
    lut.print_table('bin')
    lut.print_table('hex')
    lut.print_table('dec')
    
    # find a set of parameters
    a = 2
    c = 1
    b = 1.2
    
    # quantize thse values
    num_bits_quant = 10
    a_quant = int(2 ** (num_bits_quant-1) * a)
    b_quant = int(2 ** (num_bits_quant-1) * b)
    c_quant = int(2 ** (num_bits_quant-1) * c)
    
    row_index, angle_deg, angle_rad, sin_val, cos_val, sin_val_quant, cos_val_quant = lut.find_angle(a_quant, b_quant, c_quant)
    
    true_angle = 0.5 * np.arctan(2*b/(a-c)) if a != c else np.pi/4
    
    print('\n')
    print(f'true angle: {true_angle} radians, angle recovered from lookup table: {angle_rad} radians\n')
    print(f'true angle: {true_angle * 180 /np.pi: 0.3f} degrees, angle recovered from lookup table: {angle_rad * 180 / np.pi : 0.3f} degrees\n')
    print(f'true SIN val: {np.sin(true_angle)}, SIN val in lookup table: {sin_val}\n')
    print(f'true COS val: {np.cos(true_angle)}, COS val in lookup table: {cos_val}\n')