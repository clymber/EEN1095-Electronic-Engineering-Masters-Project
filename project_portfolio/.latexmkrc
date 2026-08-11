# Keep all generated files, including SyncTeX, in build/.
$out_dir = 'build';
$aux_dir = 'build';
$pdflatex = 'pdflatex -synctex=1 %O %S';

# Keep the exported portfolio filename stable across command-line builds.
$jobname = 'A00049113_EEN1095_Project_Portfolio';
