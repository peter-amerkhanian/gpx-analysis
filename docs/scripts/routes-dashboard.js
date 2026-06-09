window.addEventListener("load", function () {
  const sortSelect = document.getElementById("mobile-route-sort");
  const bartSelect = document.getElementById("mobile-route-bart");
  const gravelRadios = Array.from(
    document.querySelectorAll('input[name="mobile-route-gravel"]'),
  );
  const grid = document.getElementById("mobile-route-grid");
  if (!sortSelect || !bartSelect || !grid) {
    return;
  }

  const sorters = {
    miles_asc: (a, b) => Number(a.dataset.miles) - Number(b.dataset.miles),
    miles_desc: (a, b) => Number(b.dataset.miles) - Number(a.dataset.miles),
    elev_asc: (a, b) => Number(a.dataset.elevation) - Number(b.dataset.elevation),
    elev_desc: (a, b) => Number(b.dataset.elevation) - Number(a.dataset.elevation),
    time_asc: (a, b) => Number(a.dataset.time) - Number(b.dataset.time),
    time_desc: (a, b) => Number(b.dataset.time) - Number(a.dataset.time),
  };

  function setSharedMinHeight(elements) {
    elements.forEach((element) => {
      element.style.minHeight = "";
    });
    const maxHeight = Math.max(
      0,
      ...elements.map((element) => element.getBoundingClientRect().height),
    );
    elements.forEach((element) => {
      element.style.minHeight = `${maxHeight}px`;
    });
  }

  function equalizeCardTextHeights() {
    const cards = Array.from(grid.querySelectorAll(".mobile-route-card"));
    const cardStates = cards.map((card) => ({
      display: card.style.display,
      visibility: card.style.visibility,
    }));
    cards.forEach((card) => {
      card.style.display = "";
      card.style.visibility = "hidden";
    });

    setSharedMinHeight(
      cards
        .map((card) => card.querySelector(".mobile-route-title"))
        .filter(Boolean),
    );
    setSharedMinHeight(
      cards
        .map((card) => card.querySelector(".mobile-route-metrics p:first-child"))
        .filter(Boolean),
    );

    cards.forEach((card, index) => {
      card.style.display = cardStates[index].display;
      card.style.visibility = cardStates[index].visibility;
    });
  }

  function applyMobileControls() {
    const cards = Array.from(grid.querySelectorAll(".mobile-route-card"));
    const bart = bartSelect.value;
    const gravelMode =
      gravelRadios.find((radio) => radio.checked)?.value || "include";
    cards.forEach((card) => {
      const matchesBart = !bart || card.dataset.bart === bart;
      const matchesGravel =
        gravelMode === "include" || card.dataset.hasGravel !== "true";
      const visible = matchesBart && matchesGravel;
      card.style.display = visible ? "" : "none";
    });
    const sorter = sorters[sortSelect.value] || sorters.miles_asc;
    cards.sort(sorter);
    cards.forEach((card) => grid.appendChild(card));
  }

  sortSelect.addEventListener("change", applyMobileControls);
  bartSelect.addEventListener("change", applyMobileControls);
  gravelRadios.forEach((radio) => {
    radio.addEventListener("change", applyMobileControls);
  });
  window.addEventListener("resize", function () {
    equalizeCardTextHeights();
    applyMobileControls();
  });
  equalizeCardTextHeights();
});
