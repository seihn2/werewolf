from wolfplay.cli import build_parser


def test_cli_exposes_latent_cfr_and_iterative_commands(tmp_path):
    parser = build_parser()

    latent = parser.parse_args(
        ["build-latent", "--input", str(tmp_path / "games.jsonl"), "--output", "latent.json"]
    )
    cfr = parser.parse_args(
        [
            "train-deep-cfr",
            "--latent-space",
            "latent.json",
            "--output-dir",
            "cfr",
            "--iterations",
            "3",
        ]
    )
    iterative = parser.parse_args(
        [
            "iterate-policy",
            "--output-dir",
            "runs",
            "--cfr-iterations",
            "2",
        ]
    )

    assert latent.command == "build-latent"
    assert cfr.iterations == 3
    assert iterative.cfr_iterations == 2
