import timeit
import torch


def create_timing_test(size=1000, num_runs=1000):
    # Create random matrix
    m = torch.randn(size, size, device="cuda" if torch.cuda.is_available() else "cpu")

    # Define the operations
    def op1():
        return (m.pow(2)).sum()

    def op2():
        return (m.T @ m).sum()

    # Time both operations
    time1 = timeit.timeit(op1, number=num_runs)
    time2 = timeit.timeit(op2, number=num_runs)

    # Print results
    print(f"Matrix size: {size}x{size}")
    print(f"Number of runs: {num_runs}")
    print(f"m.pow(2).sum() time: {time1:.4f} seconds")
    print(f"(m.T @ m).sum() time: {time2:.4f} seconds")
    print(f"Ratio (op2/op1): {time2/time1:.2f}x slower")

    # Verify they give same result
    result1 = op1()
    result2 = op2()
    print(f"\nResults match: {torch.allclose(result1, result2)}")
    print(f"Absolute difference: {torch.abs(result1 - result2)}")


# Test with different sizes
for size in [10, 100, 1000]:
    print("\n" + "=" * 50)
    create_timing_test(size, num_runs=100)
