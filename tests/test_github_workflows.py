import base64
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import pytest
from git import Repo

from src.code_analyzer import CodeAnalyzer
from src.config import config
from src.github_client import GitHubClient
from src.main import OpentirOrchestrator


@pytest.fixture
def local_github(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("import pathlib\ndef compute(value):\n    return value + 1\n", encoding="utf-8")
    with Repo.init(source) as repository:
        with repository.config_writer() as writer:
            writer.set_value("user", "name", "Test")
            writer.set_value("user", "email", "test@example.invalid")
        repository.index.add(["main.py"])
        repository.index.commit("Initial source")
    requests = []
    date = "2026-01-01T00:00:00Z"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            path = urlparse(self.path).path
            requests.append(path)
            if "/disconnect/" in path:
                self.connection.close()
                return
            base = f"http://127.0.0.1:{self.server.server_port}"
            repository = {
                "name": "sample", "full_name": "palantir/sample", "id": 1,
                "url": base + "/repos/palantir/sample", "html_url": base + "/sample",
                "clone_url": str(source), "description": "Sample", "language": "Python",
                "stargazers_count": 2, "forks_count": 1, "size": 1, "topics": [],
                "created_at": date, "updated_at": date, "archived": False,
                "private": False, "open_issues_count": 0,
                "owner": {"login": "palantir", "id": 1, "url": base + "/users/palantir"},
                "license": {"name": "Apache License 2.0", "key": "apache-2.0", "url": base + "/licenses/apache-2.0"},
            }
            def contents(name, text):
                return {"type": "file", "name": name, "path": name, "sha": "a" * 40,
                        "size": len(text), "encoding": "base64",
                        "content": base64.b64encode(text.encode()).decode()}
            routes = {
                "/rate_limit": {"resources": {"core": {"limit": 5000, "remaining": 5000, "reset": 2000000000}},
                                "rate": {"limit": 5000, "remaining": 5000, "reset": 2000000000}},
                "/orgs/palantir": {"login": "palantir", "name": "Example organization", "id": 1,
                                  "type": "Organization", "public_repos": 1, "created_at": date,
                                  "updated_at": date, "url": base + "/orgs/palantir"},
                "/orgs/palantir/repos": [repository],
                "/repos/palantir/sample": repository,
                "/repos/palantir/sample/topics": {"names": ["example"]},
                "/repos/palantir/sample/readme": contents("README.md", "# Sample"),
                "/repos/palantir/sample/contents/requirements.txt": contents("requirements.txt", "requests>=2\n"),
                "/repos/palantir/sample/contents/package.json": contents("package.json", '{"dependencies":{"example":"1"}}'),
                "/repos/palantir/sample/contents/": [contents("main.py", "print(1)"), {"type": "dir", "name": "src", "path": "src", "sha": "b" * 40}],
                "/repos/palantir/sample/contents/src": [contents("nested.py", "print(2)")],
                "/repos/palantir/sample/languages": {"Python": 42},
                "/repos/palantir/sample/contributors": [{"login": "test", "contributions": 1}],
                "/repos/palantir/sample/releases": [{"tag_name": "v1"}],
                "/repos/palantir/sample/pulls": [],
                "/repos/palantir/sample/commits": [{"sha": "a" * 40, "commit": {"message": "Initial\nbody", "author": {"name": "Test", "date": date}}}],
            }
            routes["/repos/palantir/bare"] = dict(repository, name="bare", full_name="palantir/bare", url=base + "/repos/palantir/bare", license=None)
            routes["/repos/palantir/bare/topics"] = {"names": []}
            status = 401 if path == "/orgs/unauthorized" else 200 if path in routes else 404
            payload = routes.get(path, {"message": "Unknown endpoint"})
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    previous_url = config.github_api_url
    config.github_api_url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield config.github_api_url, requests
    finally:
        config.github_api_url = previous_url
        server.shutdown()
        server.server_close()
        thread.join()


def cli(tmp_path, api_url, *arguments):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"base_dir": str(tmp_path), "github_api_url": api_url, "github_token": "local-test"}), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "src.cli", "-c", str(settings), *arguments],
        capture_output=True, text=True, check=False, timeout=60,
    )


def test_cli_fetch_clone_analyze_docs_status_and_cleanup(tmp_path, local_github):
    api_url, requests = local_github
    output = tmp_path / "repository-list.json"
    result = cli(tmp_path, api_url, "fetch-repos", "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_text())["total_count"] == 1
    result = cli(tmp_path, api_url, "clone-all")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "repos/all_repos/sample/main.py").exists()
    result = cli(tmp_path, api_url, "clone-all", "--force")
    assert result.returncode == 0, result.stdout + result.stderr
    result = cli(tmp_path, api_url, "analyze")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "analysis_results/all_analyses.json").read_text())[0]["total_files"] == 1
    result = cli(tmp_path, api_url, "generate-docs")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "docs/api_reference/methods/index.md").is_file()
    result = cli(tmp_path, api_url, "status")
    assert result.returncode == 0, result.stdout + result.stderr
    result = cli(tmp_path, api_url, "multi-analysis", "--force", "--output-dir", str(tmp_path / "reports"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "reports/cross_repo_analysis.json").read_text())["total_repositories"] == 1
    result = cli(tmp_path, api_url, "cleanup", "--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "repos/all_repos/sample").exists()
    assert "/orgs/palantir/repos" in requests


def test_cli_build_complete_produces_documentation(tmp_path, local_github):
    result = cli(tmp_path, local_github[0], "build-complete")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "docs/index.md").is_file()


def test_github_adapter_reads_real_metadata_and_errors(local_github):
    client = GitHubClient("local-test")
    with client.github:
        details = client.get_repository_detailed_info("sample")
        assert details["readme"] == "# Sample"
        assert details["dependencies"]["requirements.txt"] == ["requests"]
        assert details["recent_commits"][0]["message"] == "Initial"
        assert details["license"]["key"] == "apache-2.0"
        assert details["file_structure"]["src"]["files"] == ["nested.py"]
        bare = client.get_repository_detailed_info("bare")
        assert bare["readme"] == ""
        assert bare["license"] == {}
        assert client.get_repository_detailed_info("absent")["error"] == "not_found"
        assert client.get_organization_info("unauthorized")["error"] == "authentication_failed"
        assert client.get_all_repositories("absent") == []


@pytest.mark.asyncio
async def test_async_github_adapter_reads_real_details_and_missing_repository(local_github):
    async with GitHubClient("local-test") as client:
        result = await client.fetch_all_repositories_with_details()
        assert result[0]["detailed_info"]["languages"] == {"Python": 42}
        assert result[0]["detailed_info"]["contributors"][0]["login"] == "test"
        assert result[0]["detailed_info"]["releases"][0]["tag_name"] == "v1"
        assert await client.get_repository_languages_async("absent") == {}
        assert await client.get_repository_contributors_async("absent") == []
        assert await client.get_repository_releases_async("absent") == []
        assert await client.get_repository_content_async("absent") == {}
        assert "/repos/palantir/absent/languages" in local_github[1]
        assert "/repos/palantir/absent/contents/" in local_github[1]


@pytest.mark.asyncio
async def test_async_metadata_reports_disconnected_http_server(local_github):
    async with GitHubClient() as client:
        assert await client.get_repository_languages_async("disconnect") == {}
        assert await client.get_repository_contributors_async("disconnect") == []
        assert await client.get_repository_releases_async("disconnect") == []
        assert await client.get_repository_content_async("disconnect") == {}
    assert "/repos/palantir/disconnect/languages" in local_github[1]


@pytest.mark.asyncio
async def test_real_orchestrator_runs_complete_workflow(tmp_path, local_github):
    orchestrator = OpentirOrchestrator(tmp_path, "local-test")
    result = await orchestrator.execute_complete_workflow()
    assert result["success"], result
    assert (tmp_path / "analysis_results/all_analyses.json").is_file()
    assert (tmp_path / "docs/index.md").is_file()


@pytest.mark.parametrize("filename,source,name,language", [
    ("main.js", 'import "module";\nfunction calculate(a) { return a; }\nclass Example {}', "calculate", "javascript"),
    ("main.ts", 'import "module";\nfunction calculate(a) { return a; }\nclass Example {}', "calculate", "typescript"),
    ("Main.java", 'import java.util.List;\npublic class Example {\npublic int calculate(int a) { return a; }\n}', "calculate", "java"),
    ("main.go", 'package main\nimport "fmt"\nfunc Calculate(a int) int { return a }\ntype Example struct {}', "Calculate", "go"),
])
def test_language_parsers_extract_actual_source(tmp_path, filename, source, name, language):
    (tmp_path / filename).write_text(source, encoding="utf-8")
    result = CodeAnalyzer().analyze_repository(tmp_path)
    assert result.languages == {language: 1}
    assert name in [element.name for element in result.all_elements]
    assert result.file_analyses[0].dependencies


@pytest.mark.parametrize("arguments", [
    ("analyze", "--repo-path", "nonexistent-repository"),
    ("analyze",),
    ("generate-docs", "--offline"),
    ("multi-analysis", "--force"),
    ("cleanup", "--yes"),
])
def test_cli_error_paths_return_nonzero(tmp_path, arguments):
    result = cli(tmp_path, "http://127.0.0.1:1", *arguments)
    assert result.returncode != 0
    assert result.stderr


def test_generate_docs_rejects_malformed_saved_analysis(tmp_path):
    output = tmp_path / "analysis_results"
    output.mkdir()
    (output / "all_analyses.json").write_text('[{"repo_name":"bad"}]', encoding="utf-8")
    (output / "functionality_matrix.json").write_text('{}', encoding="utf-8")
    result = cli(tmp_path, "http://127.0.0.1:1", "generate-docs", "--offline")
    assert result.returncode != 0
    assert not (tmp_path / "docs/index.md").exists()
