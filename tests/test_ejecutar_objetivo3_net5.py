from ejecutar_objetivo3_net5 import (
    construir_comando_oficial,
    ruta_checkpoint_ultimo,
    ruta_resumen_oficial,
)


def test_comando_oficial_no_usa_limites_y_exige_trazabilidad(tmp_path):
    comando = construir_comando_oficial(
        python="python",
        resolucion=64,
        workers=8,
        epochs=200,
        patience=20,
        batch_size=1,
        lr=0.001,
        lotes_perfil=3,
        data_root=tmp_path / "data",
        particion_manifest=tmp_path / "particion.json",
        checkpoints_dir=tmp_path / "checkpoints",
        logs_dir=tmp_path / "logs",
        resultados_dir=tmp_path / "resultados",
        reanudar=False,
    )

    assert comando[comando.index("--resolucion") + 1] == "64"
    assert comando[comando.index("--num-workers") + 1] == "8"
    assert comando[comando.index("--epochs") + 1] == "200"
    assert "--exigir-git-limpio" in comando
    assert "--exigir-git-publicado" in comando
    assert "--tag" not in comando
    assert not any(argumento.startswith("--limite") for argumento in comando)
    assert "--reanudar" not in comando


def test_comando_activa_reanudacion_solo_cuando_se_solicita(tmp_path):
    kwargs = {
        "python": "python",
        "resolucion": 32,
        "workers": 8,
        "epochs": 200,
        "patience": 20,
        "batch_size": 1,
        "lr": 0.001,
        "lotes_perfil": 3,
        "data_root": tmp_path / "data",
        "particion_manifest": tmp_path / "particion.json",
        "checkpoints_dir": tmp_path / "checkpoints",
        "logs_dir": tmp_path / "logs",
        "resultados_dir": tmp_path / "resultados",
    }

    assert "--reanudar" in construir_comando_oficial(**kwargs, reanudar=True)
    assert "--reanudar" not in construir_comando_oficial(
        **kwargs, reanudar=False,
    )


def test_rutas_oficiales_son_deterministas(tmp_path):
    assert ruta_resumen_oficial(tmp_path, 32).name == (
        "resumen_net5_octree_R32.json"
    )
    assert ruta_checkpoint_ultimo(tmp_path, 64).name == (
        "net5_octree_ultimo_R64.pth"
    )
