from pnmi import analyze_all_dummy_datasets


def main():
    results = analyze_all_dummy_datasets()
    ordered_names = ['perfect', 'high', 'medium', 'low', 'none']

    for name in ordered_names:
        result = results[name]
        print(name)
        print('  valid frames:', result['valid_frame_count'])
        print('  pnmi:', round(result['pnmi'], 6))
        print('  phone purity:', round(result['phone_purity'], 6))
        print('  cluster purity:', round(result['cluster_purity'], 6))
        print('  mutual information:', round(result['mutual_information'], 6))


if __name__ == '__main__':
    main()
