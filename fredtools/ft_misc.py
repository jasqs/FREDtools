def mergePDF(PDFFileNames, mergedPDFFileName, removeSource=False, displayInfo=False):
    """Merge multiple PDF files to a single PDF.

    The function merges multiple PDF files given as a list of
    path strings to a single PDF.

    Parameters
    ----------
    PDFFileNames : list of strings
        List of path strings to PDF files to be merged.
    mergedPDFFileName : string
        Path string where the merged PDF will be saved.
    removeSource : bool, optional
        Determine if the source PDF files should be
        removed after merge. (def. False)
    displayInfo : bool, optional
        Displays a summary of the function results. (def. False)

    Returns
    -------
    mergedPDFFileName
        Absolute path string where the merged PDF was be saved.
    """
    import fitz  # from pymupdf
    import os
    import fredtools as ft

    # check if it is a single string
    if isinstance(PDFFileNames, str):
        PDFFileNames = [PDFFileNames]

    # check if all files to be merged exist
    for PDFFileName in PDFFileNames:
        if not os.path.exists(PDFFileName):
            raise FileNotFoundError(f"The file {PDFFileName} dose not exist.")

    mergedPDF = fitz.open()

    for PDFFileName in PDFFileNames:
        with fitz.open(PDFFileName) as mfile:
            mergedPDF.insert_pdf(mfile)

    if removeSource:
        for PDFFileName in PDFFileNames:
            os.remove(PDFFileName)

    mergedPDF.save(mergedPDFFileName)

    if displayInfo:
        print(f"### {ft._currentFuncName()} ###")
        print(f"# Merged PDF files:\n# " + "\n# ".join(PDFFileNames))
        print(f"# Saved merged PDF to: ", mergedPDFFileName)
        if removeSource:
            print(f"# Removed the source PDF files")
        print("#" * len(f"### {ft._currentFuncName()} ###"))

    return os.path.abspath(mergedPDFFileName)
