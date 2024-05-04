from ultralytics.utils.benchmarks import benchmark


def main():
    # Benchmark on GPU
    benchmark(model=".\\models\\exclude_zero_scores_w_rotate_30_epochs\\weights\\best.pt",
              data='Amgad2019_data.yaml', imgsz=512, half=False, device=0)


if __name__ == "__main__":
    main()
