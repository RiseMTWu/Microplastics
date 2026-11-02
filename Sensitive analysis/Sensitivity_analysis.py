"""Run the three supplied 2019 sensitivity schemes."""

import argparse
import importlib
import os
from pathlib import Path

SCHEMES = {
    1: 'Scheme_1_Mild_tilt',
    2: 'Scheme_2_Moderate_tilt',
    3: 'Scheme_3_Strong_tilt',
}
STAGES = ('manufacturing', 'pms', 'twp', 'waste')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all',) + STAGES, default='all')
    parser.add_argument('--schemes', type=int, choices=(1, 2, 3), nargs='+', default=[1, 2, 3])
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.draws < 2:
        parser.error('Use at least two Monte Carlo draws.')
    os.environ['MC_DRAWS'] = str(args.draws)
    stages = STAGES if args.stage == 'all' else (args.stage,)
    for stage in stages:
        model = importlib.import_module(f'models.{stage}_sensitivity')
        for number in dict.fromkeys(args.schemes):
            scheme = SCHEMES[number]
            config = model.SCHEMES[scheme]
            if not Path(config['input']).is_file():
                raise FileNotFoundError(config['input'])
            if args.validate_only:
                import pandas as pd
                workbook = pd.ExcelFile(config['input'], engine='openpyxl')
                expected = ['prod-manu'] if stage == 'manufacturing' else ['Data'] if stage == 'twp' else ['Landfill', 'Incineration', 'Recycling', 'Mismanagement'] if stage == 'waste' else ['Sheet1']
                if not set(expected).issubset(workbook.sheet_names):
                    raise ValueError(f'{scheme}/{stage}: unexpected worksheets {workbook.sheet_names}')
                print(f'Validated {scheme}/{stage}: {config["input"].name}')
                continue
            summary = model.run_scheme(scheme, config['input'], config['output'])
            print(summary)


if __name__ == '__main__':
    main()
