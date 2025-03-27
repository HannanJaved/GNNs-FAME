def calculate_fairness(data, predictions, sensitive_attribute, verbose=True):
    """
    Calculate various fairness metrics.

    Args:
    label: Actual labels (binary).
    predictions: Model predictions (binary).
    sens_attr: Binary sensitive attribute for fairness evaluation.
    verbose: If True, prints debug information.

    Returns:
    A dictionary containing SPD, EOD, OAED, and TED values.
    """
    data = data.to('cpu')
    labels = data.y

    predictions = predictions.to('cpu').float()
    labels = labels.to('cpu').float()
    sensitive_attribute = sensitive_attribute.to('cpu').float()

    if verbose:
        print(f"Number of actual labels: {len(labels)}")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Number of sensitive attributes: {len(sensitive_attribute)}")
        print(f"Labels distribution: {labels.long().bincount()}")
        print(f"Predictions distribution: {predictions.long().bincount()}")
        print(f"Sensitive attribute distribution: {sensitive_attribute.long().bincount()}")

    def statistical_parity_difference():
        prob_group_1 = predictions[sensitive_attribute == 1].mean()
        prob_group_0 = predictions[sensitive_attribute == 0].mean()
        return abs(prob_group_1 - prob_group_0), prob_group_0, prob_group_1

    def equal_opportunity_difference():
        tpr_group_1 = predictions[(labels == 1) & (sensitive_attribute == 1)].mean()
        tpr_group_0 = predictions[(labels == 1) & (sensitive_attribute == 0)].mean()
        return abs(tpr_group_1 - tpr_group_0), tpr_group_0, tpr_group_1

    def overall_accuracy_equality_difference():
        acc_group_1 = (predictions[sensitive_attribute == 1] == labels[sensitive_attribute == 1]).float().mean()
        acc_group_0 = (predictions[sensitive_attribute == 0] == labels[sensitive_attribute == 0]).float().mean()
        return abs(acc_group_1 - acc_group_0), acc_group_0, acc_group_1

    def treatment_equality_difference():
        fn_group_1 = ((predictions == 0) & (labels == 1) & (sensitive_attribute == 1)).sum()
        fp_group_1 = ((predictions == 1) & (labels == 0) & (sensitive_attribute == 1)).sum()

        fn_group_0 = ((predictions == 0) & (labels == 1) & (sensitive_attribute == 0)).sum()
        fp_group_0 = ((predictions == 1) & (labels == 0) & (sensitive_attribute == 0)).sum()

        ratio_group_1 = fn_group_1 / fp_group_1 if fp_group_1 != 0 else float('inf')
        ratio_group_0 = fn_group_0 / fp_group_0 if fp_group_0 != 0 else float('inf')

        return abs(ratio_group_1 - ratio_group_0), ratio_group_0, ratio_group_1

    # Calculating each fairness metric
    spd, sp_g0, sp_g1 = statistical_parity_difference()
    eod, eod_g0, eod_g1 = equal_opportunity_difference()
    oaed, oaed_g0, oaed_g1 = overall_accuracy_equality_difference()
    ted, ted_g0, ted_g1 = treatment_equality_difference()

    return {
        'Statistical Parity Difference': spd,
        'Statistical Parity Group with S=0': sp_g0,
        'Statistical Parity Group S=1': sp_g1,
        'Equal Opportunity Difference': eod,
        'Equal Opportunity Group with S=0': eod_g0,
        'Equal Opportunity Group S=1': eod_g1,
        'Overall Accuracy Equality Difference': oaed,
        'Overall Accuracy Group with S=0': oaed_g0,
        'Overall Accuracy Group S=1': oaed_g1,
        'Treatment Equality Difference': ted,
        'Treatment Equality Group with S=0': ted_g0,
        'Treatment Equality Group S=1': ted_g1
    }