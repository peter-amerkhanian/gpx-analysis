window.addEventListener("load", function () {
  const sortSelect = document.getElementById("mobile-route-sort");
  const bartSelect = document.getElementById("mobile-route-bart");
  const grid = document.getElementById("mobile-route-grid");
  if (!sortSelect || !bartSelect || !grid) {
    return;
  }

  const sorters = {
    miles_asc: (a, b) => Number(a.dataset.miles) - Number(b.dataset.miles),
    miles_desc: (a, b) => Number(b.dataset.miles) - Number(a.dataset.miles),
    elev_asc: (a, b) => Number(a.dataset.elevation) - Number(b.dataset.elevation),
    elev_desc: (a, b) => Number(b.dataset.elevation) - Number(a.dataset.elevation),
  };

  function applyMobileControls() {
    const cards = Array.from(grid.querySelectorAll(".mobile-route-card"));
    const bart = bartSelect.value;
    cards.forEach((card) => {
      const visible = !bart || card.dataset.bart === bart;
      card.style.display = visible ? "" : "none";
    });
    const sorter = sorters[sortSelect.value] || sorters.miles_asc;
    cards.sort(sorter);
    cards.forEach((card) => grid.appendChild(card));
  }

  sortSelect.addEventListener("change", applyMobileControls);
  bartSelect.addEventListener("change", applyMobileControls);
});
