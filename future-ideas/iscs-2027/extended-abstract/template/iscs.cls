% This style file needs LaTeX version 2e
\NeedsTeXFormat{LaTeX2e}

% Style name
\ProvidesPackage{iscs}[2026/02/05]

\LoadClass[12pt,a4paper]{article}

%% Fonts
\RequirePackage{lmodern}
\RequirePackage{amsmath, amsfonts, amssymb, bm}
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}

%% Bibliography
\bibliographystyle{plain}
\RequirePackage[makeindex]{imakeidx}

%% graphics
\RequirePackage{tikz}
\RequirePackage{graphicx}

%%% geometry
\RequirePackage[a4paper,%
% showframe,%
includeheadfoot,%
textwidth=17cm,% 
textheight=25cm,% 
nomarginpar,% We don't want any margin paragraphs
% top=2cm,left=2cm,right=2cm, bottom=2cm,% mieux vaut utiliser textwidth et textheight avec includeheadfoot
headheight=0.6cm,%
footskip=1cm,
headsep=0.4cm,
]{geometry}

\makeatletter

%% Headers and footers
\RequirePackage{fancyhdr}

\pagestyle{fancy}
\lhead{\textit{\@authorsrunning}}\chead{}\rhead{\textit{\@titlerunning}}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\footrulewidth}{0.5pt}
\lfoot{\textbf{\small{International Symposium on Complex Systems}}}\cfoot{}\rfoot{\thepage}

\fancypagestyle{firststyle}
{
    \lhead{}
    \chead{}
    \rhead{}
    \renewcommand{\headrulewidth}{0pt} % removes horizontal header line
}

% Running title and authors, address
\renewcommand{\title}[1]{\def\@title{#1}}
\newcommand{\titlerunning}[1]{\def\@titlerunning{#1}}
\newcommand{\authors}[1]{\def\@authors{#1}}
\newcommand{\authorsrunning}[1]{\def\@authorsrunning{#1}}
\newcommand{\addresses}[1]{\def\@addresses{#1}}
\newcommand{\speaker}[1]{#1\textsuperscript{\checkmark}}

% Abstract and keywords
\def\@abstract{\normalsize\textbf{Abstract. }} \def\end@abstract{\par}
\def\@keywords{\normalsize\textbf{Keywords. }} \def\end@keywords{\par}
\newtoks\abstract
\newtoks\keywords
\def\cleararticleinfo{\global\abstract={}}
\def\cleararticleinfo{\global\keywords={}}

% Make title
\renewcommand{\maketitle}{
\newsavebox{\foo}
\savebox{\foo}{% <- ce '%' est important
\thispagestyle{firststyle}%
\begin{minipage}[t]{\textwidth}
  % Logo

	% Title and authors
  \begin{minipage}[c][][c]{\textwidth}
		%\begin{center}
    \par\vspace{1cm}
		\Large\bfseries\@title
		\par\vspace{0.8cm}
		\normalfont\normalsize\noindent\@authors
		% \par\vspace{1cm}
		%\end{center}
	\end{minipage}
	% Affiliations
	\par\vspace{1cm}\itshape\small
	\@addresses\\
    \textsuperscript{\checkmark} Presenting author
	\par\vspace{1em}\noindent\rule{\linewidth}{.5pt}
	% Abstract
	\normalfont
	\par\vspace{1em}
	\begin{@abstract}\the\abstract\end{@abstract}
	% Keywords
        \par\vspace{1em}
	\itshape
	\begin{@keywords}\the\keywords\end{@keywords}
	\par\vspace{1em}\noindent\rule{\textwidth}{.5pt}
\end{minipage}
}
\par\noindent\usebox{\foo}\vspace{-0.25cm}}




%% Space between sections
\usepackage{titlesec}
\titlespacing*{\section}{0pt}{0.5\baselineskip}{0.2\baselineskip}
\titlespacing*{\subsection}{0pt}{0.4\baselineskip}{0.1\baselineskip}
\titlespacing*{\subsubsection}{0pt}{0.1\baselineskip}{0\baselineskip}

% Space between paragraphs
\setlength{\parskip}{0.5\baselineskip}

% Do not indent the 1st line of a paragraph.
\setlength{\parindent}{0mm}

% For making the biblipgraphy more compact
\renewcommand{\itemsep}{\skip0}
\renewcommand{\parsep}{\skip0}



\makeatother

\endinput
