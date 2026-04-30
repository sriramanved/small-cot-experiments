from scripts.plot_style import format_iteration_k, get_method_style, metric_display_label


def test_method_style_aliases_share_publication_colors():
    assert get_method_style("Offline BC").color == get_method_style("LogLossBC").color
    assert get_method_style("TM OPD").color == get_method_style("OPD").color
    assert get_method_style("NAIL-forward").color == get_method_style("NAIL-forward, greedy rollout").color
    assert get_method_style("NAIL-OPD MC").color == get_method_style("NAIL-forward").color
    assert get_method_style("NAIL-reverse MC").color == get_method_style("NAIL-reverse, greedy rollout").color


def test_sampled_rollout_variants_are_dashed():
    assert get_method_style("NAIL-forward, sampled rollout").linestyle == "--"
    assert get_method_style("NAIL-reverse, sampled rollout").linestyle == "--"
    assert get_method_style("TM OPD, sampled rollout").linestyle == "--"


def test_paper_method_labels_and_line_styles():
    assert get_method_style("Offline BC").label == "Offline BC"
    assert get_method_style("Offline BC").color == "#000000"
    assert get_method_style("Offline BC").linestyle == ":"
    assert get_method_style("TM OPD").label == "OPD-R"
    assert get_method_style("TM OPD").linestyle == "--"
    assert get_method_style("NAIL-forward, sampled rollout").label == "OPD-F"
    assert get_method_style("NAIL-forward, sampled rollout").linestyle == "--"
    assert get_method_style("NAIL-forward, greedy rollout").label == "NAIL-F"
    assert get_method_style("NAIL-forward, greedy rollout").linestyle == "-"
    assert get_method_style("NAIL-reverse, greedy rollout").label == "NAIL-R"
    assert get_method_style("NAIL-reverse, greedy rollout").linestyle == "-"


def test_iteration_ticks_use_k_abbreviations():
    assert format_iteration_k(0) == "0"
    assert format_iteration_k(125000) == "125K"
    assert format_iteration_k(12500) == "12.5K"
    assert format_iteration_k(8_000_000) == "8M"


def test_metric_display_label_drops_clean_prefix_for_exact_metrics():
    assert metric_display_label("val/clean_full_exact") == "Reward"
    assert metric_display_label("val/clean_final_exact") == "Final exact"
