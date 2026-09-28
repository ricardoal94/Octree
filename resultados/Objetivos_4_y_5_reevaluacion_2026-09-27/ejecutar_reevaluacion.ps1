# Reevaluacion de los modelos ya entrenados (R32 y R64) con los octrees originales,
# siguiendo resultados/Objetivos_4_y_5_comparacion/README.md, seccion 2.2 (PR #26).
# No modifica el repositorio: trabaja sobre una copia externa de la carpeta publicada.

$ErrorActionPreference = "Stop"
$Repo    = "C:\Users\ricar\Documents\Codigos\Tesis"
$Base    = "C:\Users\ricar\Documents\Codigos\Reevaluacion_objetivos_4_5_20260927"
$Carpeta = "$Base\carpeta"
$Python  = "$Repo\venv\Scripts\python.exe"
$Log     = "$Base\registro_ejecucion.log"

function Escribir($texto) {
    # UTF-8 (Tee-Object de PowerShell 5.1 escribiria UTF-16).
    $texto | Out-File -FilePath $Log -Append -Encoding utf8
    Write-Output $texto
}
function Paso($titulo, $argumentos) {
    Escribir ""
    Escribir ("=" * 78)
    Escribir "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $titulo"
    Escribir "> python $argumentos"
    Escribir ("=" * 78)
    $inicio = Get-Date
    # cmd /c evita que PowerShell 5.1 convierta stderr en registros de error.
    cmd /c "`"$Python`" $argumentos 2>&1" | ForEach-Object { Escribir $_ }
    $codigo = $LASTEXITCODE
    Escribir "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] codigo de salida: $codigo | duracion: $([math]::Round(((Get-Date) - $inicio).TotalMinutes, 2)) min"
    if ($codigo -ne 0) { throw "Fallo el paso: $titulo" }
}

Set-Location $Repo
if (Test-Path $Log) { Remove-Item $Log }

Escribir "REGISTRO DE EJECUCION - Reevaluacion objetivos 4 y 5"
Escribir "Inicio: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Escribir "Repositorio: $Repo"
Escribir "Rama: $(git branch --show-current) | commit: $(git rev-parse HEAD)"
Escribir "git status --short:"
git status --short | ForEach-Object { Escribir "  $_" }
Escribir "Carpeta evaluada (copia externa, exportada de git con LF): $Carpeta"
Escribir ""
Escribir "--- Entorno"
Escribir "Sistema: $((Get-CimInstance Win32_OperatingSystem).Caption) $((Get-CimInstance Win32_OperatingSystem).Version)"
Escribir "CPU: $((Get-CimInstance Win32_Processor).Name) | nucleos: $((Get-CimInstance Win32_Processor).NumberOfCores) | hilos: $((Get-CimInstance Win32_Processor).NumberOfLogicalProcessors)"
Escribir "RAM: $([math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)) GB"
Escribir "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total,utilization.gpu,memory.used --format=csv,noheader)"
Escribir (cmd /c "`"$Python`" -c `"import sys, torch, numpy, sklearn, numba, scipy, joblib, matplotlib; print('Python', sys.version.split()[0], '| torch', torch.__version__, '| CUDA', torch.version.cuda, torch.cuda.is_available(), '| numpy', numpy.__version__, '| scikit-learn', sklearn.__version__, '| numba', numba.__version__, '| scipy', scipy.__version__, '| joblib', joblib.__version__, '| matplotlib', matplotlib.__version__)`" 2>&1")
Escribir "Octrees R32: $((Get-ChildItem "$Repo\data\octrees_32" -Recurse -Filter *.npz).Count) archivos | R64: $((Get-ChildItem "$Repo\data\octrees_64" -Recurse -Filter *.npz).Count) archivos"

Paso "1/5 Auditoria previa (hashes de datos y modelos)" "fase4_comparacion/auditar_comparacion.py --carpeta `"$Carpeta`" --exigir-modelos"
Paso "2/5 Evaluacion completa (predicciones, tiempos, memoria, analisis)" "fase4_comparacion/evaluar_comparacion.py --carpeta `"$Carpeta`""
Paso "3/5 Auditoria posterior" "fase4_comparacion/auditar_comparacion.py --carpeta `"$Carpeta`" --exigir-modelos"
Paso "4/5 Tablas" "fase4_comparacion/tablas_comparacion.py --carpeta `"$Carpeta`""
Paso "5/5 Figuras" "fase4_comparacion/figuras_comparacion.py --carpeta `"$Carpeta`""

Escribir ""
Escribir "Fin: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
