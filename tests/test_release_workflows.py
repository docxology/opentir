import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pytest
from aiohttp import web
from git import Repo
from pydantic import TypeAdapter, ValidationError

from src.code_analyzer import CodeAnalyzer, PythonParser, RepositoryAnalysis
from src.config import OpentirConfig, config
from src.config import get_config_dict, update_config, load_config_from_file, save_config_to_file
from src.docs_generator import DocumentationGenerator
from src.github_client import GitHubClient
from src.repo_manager import RepositoryManager
from src.multi_repo_analyzer import MultiRepositoryAnalyzer, CrossRepoMetrics
from src.main import OpentirOrchestrator
from src.utils import RepositoryInfo


def run_cli(workspace: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    configuration = workspace / "settings.json"
    configuration.write_text(json.dumps({"base_dir": str(workspace)}), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "src.cli", "--config-file", str(configuration), *arguments],
        capture_output=True, text=True, check=False, timeout=60,
    )


def test_saved_analysis_generates_buildable_docs_when_offline(tmp_path):
    repository = tmp_path / "sample"
    repository.mkdir()
    (repository / "sample.py").write_text(
        'import pathlib\nclass Example:\n    def compute(self, value: int) -> int:\n'
        '        """Return a value."""\n        return value\n', encoding="utf-8",
    )
    output = tmp_path / "analysis"
    analyzed = run_cli(tmp_path, "analyze", "--repo-path", str(repository), "--output-dir", str(output))
    assert analyzed.returncode == 0, analyzed.stdout + analyzed.stderr
    data = json.loads((output / "all_analyses.json").read_text())
    restored = TypeAdapter(list[RepositoryAnalysis]).validate_python(data)
    assert restored[0].file_analyses[0].elements[1].name == "compute"
    generated = run_cli(tmp_path, "generate-docs", "--offline", "--analysis-dir", str(output), "--output-dir", str(tmp_path / "docs"))
    assert generated.returncode == 0, generated.stdout + generated.stderr
    assert "pathlib" in (tmp_path / "docs/analysis/dependencies.md").read_text()
    built = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "-f", str(tmp_path / "docs.mkdocs.yml")],
        capture_output=True, text=True, check=False, timeout=60,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    assert (tmp_path / "docs_site/index.html").is_file()


def test_cli_fails_when_configuration_is_invalid(tmp_path):
    configuration = tmp_path / "bad.json"
    configuration.write_text('{"max_concurrent_requests": 0}', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "-c", str(configuration), "status"],
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode != 0
    assert "Invalid configuration" in result.stderr


def test_analysis_fails_when_repository_is_a_file(tmp_path):
    repository = tmp_path / "source.py"
    repository.write_text("print(1)", encoding="utf-8")
    with pytest.raises(ValueError):
        CodeAnalyzer().analyze_repository(repository)


def test_parser_extracts_methods_annotations_and_all_parameter_kinds():
    code = "class Example:\n    async def run(self, a, /, *items, flag=True, **options) -> int:\n        return a\n"
    method = next(element for element in PythonParser().parse(code, "test.py") if element.name == "run")
    assert method.type == "method"
    assert method.is_async
    assert set(method.parameters) == {"self", "a", "*items", "flag", "**options"}
    assert method.return_type == "int"


def test_analysis_excludes_virtual_environments_and_external_symlinks(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "main.py").write_text("def included(): pass\n", encoding="utf-8")
    (repository / ".venv").mkdir()
    (repository / ".venv/dependency.py").write_text("def excluded(): pass\n", encoding="utf-8")
    external = tmp_path / "outside.py"
    external.write_text("def external(): pass\n", encoding="utf-8")
    (repository / "linked.py").symlink_to(external)
    analysis = CodeAnalyzer().analyze_repository(repository)
    assert [element.name for element in analysis.all_elements] == ["included"]


def test_saved_analysis_rejects_stringified_elements():
    analysis = RepositoryAnalysis("bad", 0, 0, {}, [], [], {}, {})
    data = asdict(analysis)
    data["all_elements"] = ["CodeElement(name='lost')"]
    with pytest.raises(ValidationError):
        TypeAdapter(RepositoryAnalysis).validate_python(data)


def test_documentation_templates_are_read_only(tmp_path):
    templates = Path(__file__).resolve().parents[1] / "src/templates"
    before = {path.name: path.read_bytes() for path in templates.iterdir() if path.is_file()}
    DocumentationGenerator(tmp_path / "docs")
    after = {path.name: path.read_bytes() for path in templates.iterdir() if path.is_file()}
    assert after == before


def test_settings_accepts_token_and_unrelated_dotenv_entries(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("GITHUB_TOKEN=example\nUNRELATED_APP=value\n", encoding="utf-8")
    settings = OpentirConfig(_env_file=dotenv)
    assert settings.github_token == os.environ.get("GITHUB_TOKEN", "example")


@pytest.mark.asyncio
async def test_github_languages_uses_real_http_and_closes_session(tmp_path):
    app = web.Application()
    async def languages(request):
        return web.json_response({"Python": 12})
    app.router.add_get("/repos/palantir/sample/languages", languages)
    runner = web.AppRunner(app)
    await runner.setup()
    server = web.TCPSite(runner, "127.0.0.1", 0)
    await server.start()
    port = server._server.sockets[0].getsockname()[1]
    previous_url = config.github_api_url
    try:
        config.github_api_url = f"http://127.0.0.1:{port}"
        async with GitHubClient() as client:
            result = await client.get_repository_languages_async("sample")
        assert result == {"Python": 12}
        assert client.session.closed
    finally:
        config.github_api_url = previous_url
        await runner.cleanup()


def test_local_clone_skips_and_refuses_dirty_update(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    repository = Repo.init(source)
    (source / "main.py").write_text("def value(): return 1\n", encoding="utf-8")
    repository.index.add(["main.py"])
    with repository.config_writer() as writer:
        writer.set_value("user", "name", "Test")
        writer.set_value("user", "email", "test@example.invalid")
    repository.index.commit("Initial source")
    info = RepositoryInfo("sample", "local/sample", "", str(source), str(source), "Python", 0, 0, 0, [], datetime.now(), datetime.now(), False, False)
    manager = RepositoryManager(tmp_path / "workspace")
    manager.setup_repository_structure()
    cloned = manager._clone_repository_sync(info, True)
    assert cloned["success"] and cloned["action"] == "cloned"
    skipped = manager._clone_repository_sync(info, False)
    assert skipped["action"] == "skipped"
    dirty_file = Path(cloned["path"]) / "local.txt"
    dirty_file.write_text("preserve", encoding="utf-8")
    failed = manager._clone_repository_sync(info, True)
    assert not failed["success"]
    assert dirty_file.read_text() == "preserve"


@pytest.mark.asyncio
async def test_orchestrator_awaits_coroutine_returned_by_callable(tmp_path):
    orchestrator = OpentirOrchestrator(tmp_path)
    async def work():
        return 17
    result = await orchestrator._execute_step("callable", lambda: work())
    assert result == 17
    assert orchestrator.execution_state["steps_completed"] == ["callable"]


def test_multirepository_pipeline_writes_real_reports_and_charts(tmp_path):
    repositories = tmp_path / "repos/all_repos"
    for index in range(3):
        repository = repositories / f"sample-{index}"
        repository.mkdir(parents=True)
        (repository / "source.py").write_text(
            f"import pathlib\ndef calculate_{index}(value):\n    if value:\n        return {index}\n    return -1\n",
            encoding="utf-8",
        )
    analyzer = MultiRepositoryAnalyzer(tmp_path / "repos")
    metrics = analyzer.analyze_all_repositories(force_reanalyze=True)
    assert metrics.total_repositories == 3
    assert metrics.total_code_elements == 3
    reports = analyzer.generate_comprehensive_reports(metrics)
    charts = analyzer.generate_comprehensive_visualizations(metrics)
    patterns = analyzer.extract_cross_repo_patterns(metrics)
    assert reports and charts and patterns
    for path in reports.values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0
    for path in charts.values():
        assert Path(path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert analyzer.analyze_all_repositories().total_repositories == 3


def test_documentation_rejects_repository_path_traversal(tmp_path):
    analysis = RepositoryAnalysis("../outside", 0, 0, {}, [], [], {}, {})
    generator = DocumentationGenerator(tmp_path / "docs")
    with pytest.raises(ValueError):
        generator.generate_complete_documentation([analysis], {}, {})
    assert not (tmp_path / "outside.md").exists()


def test_cli_version_does_not_create_workspace_directories(tmp_path):
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "--version"], cwd=tmp_path,
        env=environment, capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "logs").exists()
    assert not (tmp_path / "repos").exists()


@pytest.mark.parametrize("constant", [True, False])
def test_correlations_distinguish_constant_from_linear_input(tmp_path, constant):
    rankings = {"by_size": [
        {"name": str(index), "total_lines": index, "total_files": 1,
         "element_count": index, "avg_complexity": 1 if constant else index}
        for index in range(1, 4)
    ]}
    metrics = CrossRepoMetrics(3, 3, 6, 6, {}, {}, {}, rankings, {})
    analyzer = MultiRepositoryAnalyzer(tmp_path / "repos")
    result = analyzer._extract_size_complexity_correlations(metrics)
    correlation = result["correlations"]["lines_vs_complexity"]
    if constant:
        assert correlation["correlation"] is None
        assert correlation["reason"] == "constant_input"
    else:
        assert correlation["correlation"] == pytest.approx(1.0)
    json.dumps(result, allow_nan=False)


def test_configuration_roundtrip_and_invalid_update_are_atomic(tmp_path):
    original = get_config_dict()
    target = tmp_path / "configuration.json"
    try:
        update_config(base_dir=tmp_path, max_concurrent_requests=7)
        save_config_to_file(str(target))
        update_config(max_concurrent_requests=3)
        load_config_from_file(str(target))
        assert config.max_concurrent_requests == 7
        assert config.base_dir == tmp_path
        with pytest.raises(ValueError):
            update_config(base_dir=tmp_path / "wrong", max_concurrent_requests=0)
        assert config.base_dir == tmp_path
        with pytest.raises(ValueError):
            update_config(unknown_setting=1)
        assert config.max_concurrent_requests == 7
    finally:
        update_config(**original)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", [
    "get_repository_content_async", "get_repository_languages_async",
    "get_repository_contributors_async", "get_repository_releases_async",
])
async def test_async_metadata_requires_open_session(method):
    client = GitHubClient()
    with pytest.raises(RuntimeError):
        await getattr(client, method)("sample")


def test_invalid_python_source_has_no_extracted_elements():
    parser = PythonParser()
    assert parser.parse("def broken(", "broken.py") == []
    assert parser.extract_imports("def broken(") == []


def test_comment_only_repository_reports_insufficient_data(tmp_path):
    repository = tmp_path / "repos/all_repos/comments"
    repository.mkdir(parents=True)
    (repository / "source.py").write_text("# Only a comment\n", encoding="utf-8")
    analyzer = MultiRepositoryAnalyzer(tmp_path / "repos")
    metrics = analyzer.analyze_all_repositories(True)
    assert metrics.total_code_elements == 0
    assert metrics.complexity_statistics == {}
    assert analyzer.extract_cross_repo_patterns(metrics)
    for filename in analyzer.generate_comprehensive_reports(metrics).values():
        assert Path(filename).is_file()
    for filename in analyzer.generate_comprehensive_visualizations(metrics).values():
        assert Path(filename).read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("directory,setting", [("tests", "include_tests"), ("examples", "include_examples")])
def test_analyzer_honors_directory_inclusion_settings(tmp_path, directory, setting):
    (tmp_path / directory).mkdir()
    (tmp_path / directory / "code.py").write_text("def included(): pass\n", encoding="utf-8")
    previous = getattr(config, setting)
    try:
        setattr(config, setting, False)
        assert CodeAnalyzer().analyze_repository(tmp_path).total_files == 0
        setattr(config, setting, True)
        assert CodeAnalyzer().analyze_repository(tmp_path).total_files == 1
    finally:
        setattr(config, setting, previous)


def test_analyzer_skips_binary_oversized_and_unsupported_files(tmp_path):
    (tmp_path / "binary.py").write_bytes(b"\x00binary")
    (tmp_path / "source.rs").write_text("fn sample() {}", encoding="utf-8")
    with (tmp_path / "oversized.py").open("wb") as stream:
        stream.truncate(11 * 1024 * 1024)
    assert CodeAnalyzer().analyze_repository(tmp_path).total_files == 0


@pytest.mark.asyncio
async def test_retry_rejects_negative_retry_count():
    from src.utils import AsyncUtils
    async def action():
        return 1
    with pytest.raises(ValueError):
        await AsyncUtils.retry_async(action, max_retries=-1)


def test_go_parser_handles_multiple_imports():
    from src.code_analyzer import GoParser
    source = 'import (\n "fmt"\n "os"\n)\nfunc Main() {}'
    assert GoParser().extract_imports(source) == ["fmt", "os"]


def test_interactive_token_prompt_can_skip_without_persisting_token(tmp_path):
    import pty
    master, terminal = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-c", "import json; from src.cli import prompt_for_github_token; result = prompt_for_github_token(); print('RESULT=' + json.dumps(result))"],
        stdin=terminal, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])),
    )
    os.close(terminal)
    try:
        os.write(master, b"3\n")
        output, error = process.communicate(timeout=30)
    finally:
        os.close(master)
    assert process.returncode == 0, error
    assert json.loads(output.split("RESULT=")[-1]) is None
    assert not (tmp_path / ".env").exists()
