# Load the Overleaf-compatible prerequisite build locally too.
do './latexmkrc'
    or die "Could not load latexmkrc: $@ $!";

# Keep all generated files, including SyncTeX, in build/.
$out_dir = 'build';
$aux_dir = 'build';
$pdflatex = 'pdflatex -synctex=1 %O %S';
