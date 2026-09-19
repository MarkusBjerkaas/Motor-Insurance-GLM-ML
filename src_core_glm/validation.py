"""Felles foldbygging for alle GLM-notebookene (frekvens, severity og Tweedie)."""

from sklearn.model_selection import GroupKFold


def build_group_folds(
    frame, group_column="insured_id", n_splits=5, shuffle=True, random_state=100
):
    """Bygg gruppefolder på hele modellpopulasjonen.

    Samme ``frame.index``, gruppekolonne og ``random_state`` gir identiske
    folder. Frekvens, severity og Tweedie kaller derfor funksjonen med den
    *fulle* modellrammen, slik at OOF-prediksjonene kan sammenlignes rad for
    rad. Modeller som fittes på en delmengde (severity) skjærer foldene mot
    delmengdens indeks i ``cross_validate_glm``, de bygger dem ikke selv.

    Returns
    -------
    list of dict
        Én dict per fold med ``fold`` (navn), ``train_index`` og ``val_index``
        (radindekser i ``frame``).
    """
    group_kfold = GroupKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )
    splits = group_kfold.split(frame, groups=frame[group_column])
    return [
        {
            "fold": f"gruppe_{number + 1}",
            "train_index": frame.index[train_positions],
            "val_index": frame.index[val_positions],
        }
        for number, (train_positions, val_positions) in enumerate(splits)
    ]
