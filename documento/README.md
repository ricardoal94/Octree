# Trabajo de grado - proyecto Overleaf

Proyecto maestro en LaTeX para el trabajo de grado sobre clasificación 3D con
octrees y Net5.

## Carga en Overleaf

1. En Overleaf, seleccione **New Project > Upload Project**.
2. Cargue el archivo ZIP completo.
3. Verifique que `main.tex` sea el documento principal.
4. Use `pdfLaTeX` como compilador. Overleaf ejecutará Biber automáticamente
   para las referencias en estilo APA.
5. Copie el archivo original `logo_uis.pdf` en la carpeta `Figuras/`. El JPG
   incluido funciona únicamente como respaldo y puede eliminarse después.

## Organización

- `main.tex`: documento maestro y orden de inclusión.
- `settings.tex`: paquetes, formato, referencias y comandos comunes.
- `datos.tex`: título, autores, director, codirector e información institucional.
- `preliminares/`: portada, contraportada, resumen, abstract y páginas opcionales.
- `capitulos/`: un archivo independiente por capítulo.
- `anexos/`: anexos posteriores a las referencias.
- `Figuras/`: logotipo, diagramas, gráficas e imágenes.
- `referencias.bib`: base bibliográfica administrada con Biber.

## Reglas de trabajo

- Edite los datos generales únicamente en `datos.tex`.
- No cargue paquetes nuevos directamente en los capítulos; agréguelos en
  `settings.tex` para mantener una configuración única.
- Use `\label{}` y `\cref{}` para todas las referencias cruzadas.
- Incorpore las fuentes en `referencias.bib` y cite mediante `\textcite{}` o
  `\parencite{}`.
- Mantenga tablas extensas, configuraciones completas y resultados secundarios
  en los anexos cuando interrumpan la lectura del cuerpo principal.
