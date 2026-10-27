# R2-07 复现命令

在PowerShell执行。仅CPU合成输入；不安装依赖、不读取数据集、不访问validation/test/locked_test。实际本轮命令的退出码见validation.json；这些命令不能把物理BLOCKED变成PASS。解释器及包版本见runtime_environment.json。

工作目录与8项新测试：

    Set-Location D:/shen/PKU/PIJWM
    python -B -m unittest discover -s code/tests -p test_airfogsim_service_interface_r2_07.py -v

实际AirFogSim探针（两次各8步由脚本内部执行；建议用新目录保留已验收回执）：

    & D:/miniconda/envs/airfogsim/python.exe -B code/scripts/audit_airfogsim_trajectory_clock_radio_r2_07.py --output code/artifacts/audit/v2u4real_airfogsim_r2_07_reproduction

R1只回归两个轨迹合同测试，临时目录必须是本轮自有目录；不跑原生反例/完整旧环境/旧数据审计：

    @'
    import sys, unittest, tempfile, random
    from pathlib import Path
    import numpy as np
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    root=Path.cwd()
    sys.path[:0]=[str(root/'code/src'),str(root/'code/scripts'),str(root/'code/tests')]
    import test_v2u4real_interface_v1 as old
    out=root/'code/artifacts/audit/v2u4real_airfogsim_r2_07_20261027/regression_work'
    out.mkdir(exist_ok=True)
    def setup(self):
        self.tmp=tempfile.TemporaryDirectory(dir=out)
        self.out=Path(self.tmp.name)
        self.c=self.probe.prepare_config(self.config,self.out)
        random.seed(20261025)
        np.random.seed(20261025)
    old.AirFogSimTests.setUp=setup
    suite=unittest.TestSuite(old.AirFogSimTests(n) for n in [
        'test_replay_contract_refuses_irregular_time_and_outside_bounds',
        'test_replay_reset_exhaustion_and_control_rejection'])
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not r.wasSuccessful())
    '@ | & D:/miniconda/envs/airfogsim/python.exe -B -

R2-05B只回归两个应用接入测试：

    @'
    import sys, unittest
    from pathlib import Path
    root=Path.cwd()
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    sys.path[:0]=[str(root/'code/src'),str(root/'code/scripts'),str(root/'code/tests')]
    suite=unittest.defaultTestLoader.loadTestsFromName('test_v2u4real_r2_05b_interface_v1.AirFogSimIngressTests')
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not r.wasSuccessful())
    '@ | python -B -

无落盘编译与上下文/注册表检查：

    @'
    from pathlib import Path
    import sys,json,hashlib
    root=Path.cwd()
    sys.path.insert(0,str(root/'code/scripts'))
    import build_project_knowledge_index_v1 as b
    out=root/'code/artifacts/audit/v2u4real_airfogsim_r2_07_20261027'
    src=json.loads((out/'source_identity.json').read_text(encoding='utf-8'))
    for p,h in src['files'].items():
        raw=(root/p).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==h
        compile(raw,str(root/p),'exec')
    assert not b.validate_ai_context(root)['errors']
    b.validate_experiment_registry(root,json.loads((root/'docs/registries/experiment_registry.json').read_text(encoding='utf-8')))
    b.validate_question_routes(root,json.loads((root/'docs/registries/question_routes.json').read_text(encoding='utf-8')))
    print('PASS: four sources; nine context files; registry checks')
    '@ | python -B -

知识索引只能只读检查，本轮没有执行重建命令：

    python -B code/scripts/build_project_knowledge_index_v1.py --check

预期仍exit1并指出5份既有生成文件不一致；这项失败不被其他测试通过抵消，推送保持阻塞。合成诊断的重复一致性不是物理可信度证明；本轮没有多时隙传输或任务生命周期。
