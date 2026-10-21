"""substitute.config: a tool's options from a TOML file."""
import argparse

import pytest

from dace_fortran.substitute.config import parse_with_config


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--tile-fuse", action="store_true")
    ap.add_argument("--stub", action="append", default=[])
    ap.add_argument("--heavy-launch-bounds", default="128, 4")
    ap.add_argument("out")
    return ap


def test_the_file_sets_defaults_the_command_line_overrides_and_adds(tmp_path):
    cfg = tmp_path / "app.toml"
    cfg.write_text('[tool]\nstub = ["a"]\n\n[tool.gpu]\ntile_fuse = true\nheavy_launch_bounds = "64, 2"\n')
    args = parse_with_config(_parser(), "tool", ["--config", str(cfg), "--stub", "b", "out"])
    assert args.stub == ["a", "b"]
    assert not args.tile_fuse and args.heavy_launch_bounds == "128, 4"  # the [tool.gpu] table needs --gpu
    args = parse_with_config(_parser(), "tool", ["--config", str(cfg), "--gpu", "--heavy-launch-bounds", "32, 1", "out"])
    assert args.tile_fuse and args.heavy_launch_bounds == "32, 1"


def test_without_a_file_the_parser_s_own_defaults_hold():
    args = parse_with_config(_parser(), "tool", ["out"])
    assert args.stub == [] and not args.tile_fuse


def test_a_key_naming_no_option_is_an_error(tmp_path):
    cfg = tmp_path / "app.toml"
    cfg.write_text("[tool]\ntile_fuze = true\n")
    with pytest.raises(SystemExit):
        parse_with_config(_parser(), "tool", ["--config", str(cfg), "out"])
