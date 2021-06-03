FRACTIONS = sorted([1.0 / (2 ** two * 5 ** five) for two in range(0, 3) for five in range(0, 3)])

if __name__ == '__main__':
    import torch
    from sys import argv

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties('cuda:0').total_memory // 1048576
        model_size = float(argv[1])

        for fraction in FRACTIONS:
            if gpu_memory * fraction > model_size:
                print(fraction)
                exit(0)

    print(0)
