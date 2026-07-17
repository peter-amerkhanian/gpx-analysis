from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RouteLinks:
    strava_effort: str | None = None


@dataclass(frozen=True)
class RouteMedia:
    hero_image: str | None = None
    gallery: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteConfig:
    slug: str
    source: str
    title: str | None = None
    reverse: bool = False
    links: RouteLinks = field(default_factory=RouteLinks)
    media: RouteMedia = field(default_factory=RouteMedia)

    @property
    def display_title(self) -> str:
        if self.title:
            return self.title

        stem = Path(self.source).stem.replace("_", " ").replace("-", " ")
        return " ".join(part.capitalize() for part in stem.split())


def load_routes(manifest_path: Path, root: Path) -> list[RouteConfig]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    raw_routes = payload.get("routes", [])
    if not isinstance(raw_routes, list):
        raise ValueError(f"{manifest_path} must define a top-level 'routes' list")

    routes: list[RouteConfig] = []
    seen_slugs: set[str] = set()
    for index, raw_route in enumerate(raw_routes, start=1):
        if not isinstance(raw_route, dict):
            raise ValueError(f"Route entry #{index} in {manifest_path} must be a mapping")

        slug = str(raw_route.get("slug", "")).strip()
        source = str(raw_route.get("source", "")).strip()
        if not slug:
            raise ValueError(f"Route entry #{index} in {manifest_path} is missing 'slug'")
        if slug in seen_slugs:
            raise ValueError(f"Duplicate route slug '{slug}' in {manifest_path}")
        if not source:
            raise ValueError(f"Route '{slug}' in {manifest_path} is missing 'source'")

        source_path = root / source
        if not source_path.exists():
            raise FileNotFoundError(f"Route '{slug}' source does not exist: {source_path}")

        raw_links = raw_route.get("links") or {}
        if not isinstance(raw_links, dict):
            raise ValueError(f"Route '{slug}' links must be a mapping")

        raw_media = raw_route.get("media") or {}
        if not isinstance(raw_media, dict):
            raise ValueError(f"Route '{slug}' media must be a mapping")

        hero_image = raw_media.get("hero_image")
        if hero_image is not None:
            hero_image = str(hero_image).strip() or None
            if hero_image and not (root / "quarto" / hero_image).exists():
                raise FileNotFoundError(
                    f"Route '{slug}' hero image does not exist under quarto/: {hero_image}"
                )

        gallery_items = tuple(str(item).strip() for item in raw_media.get("gallery", []) if str(item).strip())
        for image_path in gallery_items:
            if not (root / "quarto" / image_path).exists():
                raise FileNotFoundError(
                    f"Route '{slug}' gallery image does not exist under quarto/: {image_path}"
                )

        routes.append(
            RouteConfig(
                slug=slug,
                source=source,
                title=(str(raw_route.get("title", "")).strip() or None),
                reverse=bool(raw_route.get("reverse", False)),
                links=RouteLinks(
                    strava_effort=(str(raw_links.get("strava_effort", "")).strip() or None),
                ),
                media=RouteMedia(
                    hero_image=hero_image,
                    gallery=gallery_items,
                ),
            )
        )
        seen_slugs.add(slug)

    return routes
