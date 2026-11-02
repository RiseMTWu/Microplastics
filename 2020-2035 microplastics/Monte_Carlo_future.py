"""Run the 2020-2035 manufacturing, waste, TWP, and microfiber models."""

import argparse
import importlib
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STAGES = ('manufacturing', 'waste', 'twp', 'mfs')


def validate_inputs(stage, model, years, scenarios):
    import pandas as pd

    files = []
    for year in years:
        if stage == 'manufacturing':
            files.append((model.INPUT_DIR / f'{year}.xlsx', ['prod-manu']))
        else:
            for scenario in scenarios:
                folder = {'waste': 'Waste', 'twp': 'TWP', 'mfs': 'MFs'}[stage]
                sheets = ['Landfill', 'Incineration', 'Recycling', 'Mismanagement'] if stage == 'waste' else ['Data']
                files.append((BASE_DIR / f'{folder}-dataset' / f'{year}-{scenario}.xlsx', sheets))
    for file, sheets in files:
        if not file.is_file():
            raise FileNotFoundError(f'Dataset not found: {file}')
        sample_ids = None
        for sheet in sheets:
            sample = pd.read_excel(file, sheet_name=sheet, nrows=2, engine='openpyxl')
            required = ['city', 'Category', 'level']
            if stage == 'manufacturing':
                required = model.required_columns
            elif stage == 'mfs':
                required += ['population', 'synthetic fibers', 'FrequencyMW', 'FrequencyHW', 'EF-MW', 'EF-HW', 'Load-MW', 'Load-HW']
            elif stage == 'twp':
                required += ['small1', 'medium1', 'big1', 'small2', 'medium2', 'big2', 'Airplaneinput', 'TCR']
            else:
                required += ['province', f'{sheet}/kt', f'{sheet}-EF']
            missing = sorted(set(required) - set(sample.columns))
            if missing:
                raise ValueError(f'{file.name}/{sheet}: missing columns {missing}')
            if stage == 'waste':
                identifiers = sample[['city', 'Category', 'level', 'province']]
                if sample_ids is not None and not sample_ids.equals(identifiers):
                    raise ValueError(f'{file.name}/{sheet}: pathway identifiers do not match')
                sample_ids = identifiers
    print(f'{stage}: validated file availability, worksheets, and required headers for {len(files)} datasets.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all',) + STAGES, default='all')
    parser.add_argument('--years', nargs='+', type=int, default=list(range(2020, 2036)))
    parser.add_argument('--scenarios', nargs='+', choices=('BAU', 'Mitigation'), default=['BAU', 'Mitigation'])
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--overwrite', action='store_true', help='Recompute existing scenario outputs.')
    parser.add_argument('--validate-only', action='store_true', help='Check datasets without running simulations.')
    args = parser.parse_args()
    if args.draws < 2 or any(year not in range(2020, 2036) for year in args.years):
        parser.error('Use at least two draws and years between 2020 and 2035.')
    os.environ['MC_DRAWS'] = str(args.draws)
    years = sorted(set(args.years))
    stages = STAGES if args.stage == 'all' else (args.stage,)
    for stage in stages:
        model = importlib.import_module(f'models.{stage}_future')
        if stage in ('manufacturing', 'waste'):
            model.YEARS = years
        else:
            model.years = years
        if stage == 'waste':
            model.SCENARIOS = args.scenarios
        elif stage in ('twp', 'mfs'):
            model.scenarios = args.scenarios
        if hasattr(model, 'SKIP_EXISTING'):
            model.SKIP_EXISTING = not args.overwrite
        validate_inputs(stage, model, years, args.scenarios)
        if not args.validate_only:
            model.main()


if __name__ == '__main__':
    main()
