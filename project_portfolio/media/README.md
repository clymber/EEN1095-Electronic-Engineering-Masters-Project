# Media

Store all figures and other image assets for the paper in this directory.

Use paths relative to this directory, for example:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{model_architecture.pdf}
    \caption{Architecture of the proposed surrogate model.}
    \label{fig:model-architecture}
\end{figure}
```

Because `main.tex` defines `\graphicspath{{media/}}`, the filename alone is sufficient.
Prefer vector PDF figures for plots and diagrams; use PNG for raster images where necessary.
