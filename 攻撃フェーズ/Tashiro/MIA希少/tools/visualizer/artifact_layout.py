"""Shared output locations for the offline visualizer documents."""
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE.parents[1] / 'docs' / 'visualizer'
LOCATIONS = {'MIA_5query_patterns.html': 'archive/MIA_5query_patterns.html', 'MIA_5query_pairs.html': 'archive/MIA_5query_pairs.html', 'MIA_5query_reference.html': 'archive/MIA_5query_reference.html', 'size_report_5query.json': 'resources/size_report_5query.json', 'size_report_5query_adaptive.json': 'resources/size_report_5query_adaptive.json', 'size_report.json': 'resources/size_report.json', 'size_report_5query_patterns.json': 'resources/size_report_5query_patterns.json', 'size_report_5query_reference.json': 'resources/size_report_5query_reference.json', 'nonadaptive_code.json': 'resources/nonadaptive_code.json', 'size_report_5query_pairs.json': 'resources/size_report_5query_pairs.json', 'MIA_4query_visualizer_bundle.zip': 'downloads/MIA_visualizer_bundle.zip'}


def artifact_path(name):
    return DEST / LOCATIONS.get(name, name)


def rewrite_links(html, output):
    def replace(match):
        href = match.group(1)
        if href.startswith(('https:', 'http:', '#', 'mailto:', 'blob:', 'data:')):
            return match.group(0)
        name, separator, fragment = href.partition('#')
        if '/' in name or name not in LOCATIONS and not name.startswith('MIA_'):
            return match.group(0)
        target = artifact_path(name)
        relative = os.path.relpath(target, output.parent).replace(os.sep, '/')
        return 'href="' + relative + (separator + fragment if separator else '') + '"'
    return re.sub(r'href="([^"\n]+)"', replace, html)
