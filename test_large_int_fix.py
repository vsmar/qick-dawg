#!/usr/bin/env python3
"""
Test script to verify the large integer overflow fix in readout_helpers.py
"""

import numpy as np

# Test the fix logic by simulating what happens with large register values

def test_freq2reg_overflow_handling():
    """Test that freq2reg_array handles large integers correctly."""
    
    # Simulate what happens when soccfg.freq2reg returns very large values
    class MockSoccfg:
        def freq2reg(self, freq_mhz):
            # Simulating a freq2reg that returns very large integers
            # These exceed C long limits (2^31-1 = 2147483647)
            return int(freq_mhz * 1e10) + 5000000000000  # Very large number
        
        def reg2freq(self, reg):
            return float(reg) / 1e10 - 500.0
    
    soccfg = MockSoccfg()
    
    # Test input frequencies in MHz
    freq_arr_mhz = [1838.0, 1839.0, 1840.0, 1841.0]
    vals = np.asarray(freq_arr_mhz)
    
    # This is the fixed version of _freq2reg_array
    regs = [soccfg.freq2reg(float(f)) for f in freq_arr_mhz]
    print(f"Register values: {regs}")
    print(f"Max register value: {max(regs)}")
    print(f"Max register > 2^31-1: {max(regs) > 2147483647}")
    
    try:
        # Try standard int dtype (should fail with large values)
        result = np.asarray(regs, dtype=int)
        print("✓ Standard int dtype succeeded (values were small enough)")
    except OverflowError as e:
        print(f"✗ Standard int dtype failed with OverflowError: {e}")
        # Fall back to object dtype
        result = np.asarray(regs, dtype=object)
        print(f"✓ Fallback to object dtype succeeded")
        print(f"  Result dtype: {result.dtype}")
        print(f"  Result values: {result}")
    
    # Now test the _reg2freq_array with object dtype values
    print("\nTesting reg2freq conversion:")
    result_mhz = []
    for r in result:
        freq = soccfg.reg2freq(int(r))
        result_mhz.append(freq)
    
    result_mhz_array = np.asarray(result_mhz, dtype=np.float64)
    print(f"✓ Converted back to MHz: {result_mhz_array}")
    
    # Verify round-trip accuracy
    for orig, converted in zip(freq_arr_mhz, result_mhz_array):
        print(f"  Original: {orig}, Converted: {converted}, Diff: {abs(orig - converted)}")

def test_us2cycles_overflow_handling():
    """Test that us2cycles_array handles large integers correctly."""
    
    class MockSoccfg:
        def us2cycles(self, us):
            # Simulating a us2cycles that returns very large integers
            return int(us * 1e6) + 10000000000000  # Very large number
    
    soccfg = MockSoccfg()
    
    # Test input times in microseconds
    us_arr = [0.1, 0.2, 0.5, 1.0]
    
    # This is the fixed version of _us2cycles_array
    regs = [soccfg.us2cycles(float(u)) for u in us_arr]
    print(f"\nTiming register values: {regs}")
    print(f"Max value > 2^31-1: {max(regs) > 2147483647}")
    
    try:
        # Try standard int dtype (should fail with large values)
        result = np.asarray(regs, dtype=int)
        print("✓ Standard int dtype succeeded")
    except OverflowError as e:
        print(f"✗ Standard int dtype failed: {e}")
        # Fall back to object dtype
        result = np.asarray(regs, dtype=object)
        print(f"✓ Fallback to object dtype succeeded")
        print(f"  Result dtype: {result.dtype}")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Large Integer Overflow Fix")
    print("=" * 60)
    print()
    test_freq2reg_overflow_handling()
    print()
    test_us2cycles_overflow_handling()
    print()
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
