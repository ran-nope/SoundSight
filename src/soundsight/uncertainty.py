"""Conservative ranking-margin warning, not calibrated unknown detection."""
def assess_ranking(results, minimum_margin=0.2):
    if minimum_margin < 0:
        raise ValueError('Margin must be nonnegative.')
    margin = results[0]['score'] - results[1]['score'] if len(results) >= 2 else 0.0
    uncertain = margin < minimum_margin
    return {'label': 'Uncertain / Unknown' if uncertain else results[0]['sound'],
            'uncertain': uncertain, 'margin': margin}
