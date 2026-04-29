from scripts.plot_style import get_method_style


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
