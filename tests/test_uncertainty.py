from soundsight.uncertainty import assess_ranking


def test_margin_warning_and_empty_rankings():
    assert assess_ranking([])['uncertain']
    assert assess_ranking([{'sound': 'dog', 'score': 4}, {'sound': 'rain', 'score': 3.9}])['uncertain']
    assert not assess_ranking([{'sound': 'dog', 'score': 4}, {'sound': 'rain', 'score': 3}])['uncertain']
