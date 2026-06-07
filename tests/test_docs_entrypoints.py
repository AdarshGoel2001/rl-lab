from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_points_agents_to_current_research_loop():
    text = (ROOT / "README.md").read_text()

    for phrase in [
        "executable chronology",
        "PlaNet",
        "DMC cartpole_swingup",
        "reports/world_model_runs.csv",
        "docs/repo_map.md",
        "docs/repo_inventory.md",
        "docs/research_lifecycle.md",
        "docs/agent_team_operating_model.md",
        "docs/roadmap/world_model_chronology.md",
        "docs/contracts/run_artifacts.md",
        "scripts/GPU/gpu_status.sh",
        "scripts/GPU/gpu_pull_latest.sh --run experiments/<run_name> --analyze",
    ]:
        assert phrase in text


def test_agents_is_the_single_agent_entrypoint():
    agents = ROOT / "AGENTS.md"

    assert agents.exists(), "AGENTS.md should be the single agent entrypoint."
    assert not (ROOT / "CLAUDE.md").exists(), "CLAUDE.md duplicates AGENTS.md."

    text = agents.read_text()
    for phrase in [
        "docs/roadmap/world_model_chronology.md",
        "docs/roadmap/eval_ladder.md",
        "docs/repo_map.md",
        "docs/repo_inventory.md",
        "docs/research_lifecycle.md",
        "docs/agent_team_operating_model.md",
        "docs/executable_doc_audit.md",
        "docs/contracts/run_artifacts.md",
        "scripts/GPU/gpu_status.sh",
        "scripts/GPU/gpu_sync_patch.sh --paths",
        "scripts/GPU/gpu_run.sh --session",
        "scripts/GPU/gpu_pull_latest.sh --run",
        "scripts/GPU/gpu_pull_patch.sh",
        "orchestrator owns infrastructure",
        "workflows own algorithm logic",
    ]:
        assert phrase in text

    assert len(text.splitlines()) < 180


def test_agent_team_operating_model_documents_parallel_roles():
    text = (ROOT / "docs" / "agent_team_operating_model.md").read_text()

    for phrase in [
        "Parallelize roles, not unrelated papers.",
        "Paper Agent",
        "Workflow Agent",
        "GPU Ops Agent",
        "Diagnostics Agent",
        "Manifest And Narrative Agent",
        "Handoff Packet",
        "Goal:",
        "Commands run:",
        "Open questions:",
    ]:
        assert phrase in text


def test_repo_map_marks_live_workflow_and_future_material():
    text = (ROOT / "docs" / "repo_map.md").read_text()

    for phrase in [
        "Live Research Surface",
        "docs/repo_inventory.md",
        "docs/research_lifecycle.md",
        "docs/executable_doc_audit.md",
        "docs/agent_handoff_template.md",
        "src/workflows/planet.py",
        "configs/workflow/planet.yaml",
        "Support And Future Material",
        "Do not assume their presence means the corresponding algorithm is implemented.",
        "Do not leave dead Hydra experiment entrypoints around as examples.",
    ]:
        assert phrase in text


def test_repo_inventory_classifies_active_support_future_and_stale_surfaces():
    text = (ROOT / "docs" / "repo_inventory.md").read_text()

    for phrase in [
        "active: used by the current PlaNet chapter",
        "support: tested utility code",
        "future: planned or exploratory material",
        "stale: remove, rewrite, or move into notes",
        "src/workflows/planet.py",
        "src/components/dynamics/",
        "Do not recreate those names as compatibility shims.",
    ]:
        assert phrase in text


def test_research_lifecycle_documents_paper_to_narrative_path():
    text = (ROOT / "docs" / "research_lifecycle.md").read_text()

    for phrase in [
        "paper brief -> implementation plan -> tiny validation -> GPU run",
        "research_notes/rough_notes/<chapter>_brief.md",
        "scripts/GPU/gpu_sync_patch.sh",
        "scripts/GPU/gpu_pull_latest.sh",
        "reports/world_model_runs.csv",
        "Narrative Update",
        "Review checklist",
    ]:
        assert phrase in text


def test_handoff_template_contains_required_agent_state():
    text = (ROOT / "docs" / "agent_handoff_template.md").read_text()

    for phrase in [
        "Goal:",
        "Current repo state:",
        "Files touched:",
        "Commands run:",
        "Passing tests:",
        "Remote session names:",
        "Diagnostics produced:",
        "Recommended next action:",
    ]:
        assert phrase in text


def test_executable_doc_audit_maps_claims_to_checks():
    text = (ROOT / "docs" / "executable_doc_audit.md").read_text()

    for phrase in [
        "tests/test_docs_entrypoints.py",
        "tests/test_orchestrator_evaluation_contract.py",
        "tests/test_run_artifact_contract.py",
        "tests/test_gpu_scripts_contract.py",
        "tests/test_world_model_sequence_buffer.py",
        "If a doc introduces a new durable claim",
    ]:
        assert phrase in text


def test_resume_docs_name_research_continuation_mode():
    agents_text = (ROOT / "AGENTS.md").read_text()
    buffers_text = (ROOT / "docs" / "contracts" / "buffers.md").read_text()

    for text in [agents_text, buffers_text]:
        assert "warm_start_optimizer" in text
        assert "research continuation" in text
        assert "fault tolerance" in text


def test_workflow_contract_documents_eval_boundary():
    text = (ROOT / "docs" / "contracts" / "workflow_data_contract.md").read_text()

    for phrase in [
        "orchestrator owns eval cadence",
        "workflow owns eval semantics",
        "must implement `evaluate()`",
        "Do not evaluate by routing through `collect_step`",
    ]:
        assert phrase in text


def test_component_contract_marks_planet_active_and_old_components_support():
    text = (ROOT / "docs" / "contracts" / "component_interfaces.md").read_text()

    for phrase in [
        "Active example:",
        "RSSMRepresentationLearner",
        "Support examples:",
        "No current live workflow uses these dynamics components directly.",
        "current PlaNet chapter",
    ]:
        assert phrase in text

    assert "OG World Models workflow expects" not in text
    assert "Current OG World" not in text


def test_no_stale_agent_discovery_surfaces_remain():
    stale_paths = [
        "configs/experiments/ppo_cartpole_modular.yaml",
        "configs/sweeps/ppo_extensive_sweep.yaml",
        "src/algorithms",
        "src/core",
        "src/networks",
        "src/paradigms/model_free",
        "src/workflows/world_models",
        "scripts/sweep.py",
        "scripts/analyze_sweep.py",
        "NOTES.md",
        "conversations",
    ]

    for relative in stale_paths:
        assert not (ROOT / relative).exists(), f"Remove or quarantine stale surface: {relative}"

    setup_text = (ROOT / "setup.py").read_text()
    requirements_text = (ROOT / "requirements.txt").read_text()
    config_text = (ROOT / "configs" / "config.yaml").read_text()

    assert "CLAUDE.md" not in setup_text
    assert "vizdoom" not in requirements_text.lower()
    assert "dreamer_cartpole" not in config_text
