document$.subscribe(async () => {
  const diagrams = [];
  for (const diagram of document.querySelectorAll(".mermaid")) {
    const code = diagram.querySelector(":scope > code");
    if (code) {
      const replacement = document.createElement("div");
      replacement.className = "mermaid";
      replacement.textContent = code.textContent;
      diagram.replaceWith(replacement);
      diagrams.push(replacement);
    } else {
      diagrams.push(diagram);
    }
  }
  mermaid.initialize({
    startOnLoad: false,
    theme: document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "neutral",
    securityLevel: "strict",
    flowchart: { htmlLabels: false, curve: "basis" },
  });
  try {
    await mermaid.run({ nodes: diagrams });
  } catch (error) {
    console.error("Mermaid render failed:", error?.message ?? error);
  }
});
