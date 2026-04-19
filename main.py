from experiments import (
    experiment_single,
    experiment_multiple,
    experiment_hilbert
)

if __name__ == "__main__":
    print("==========================")
    print("Эксперимент 1")
    print("==========================")
    experiment_single()

    print("\n==========================")
    print("Эксперимент 2")
    print("==========================")
    experiment_multiple()

    print("\n==========================")
    print("Эксперимент 3")
    print("==========================")
    experiment_hilbert()
