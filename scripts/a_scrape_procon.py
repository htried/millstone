import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
import re
from time import sleep
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from utils.utils import *


# Extract the main content text
def extract_main_content(soup):
    # Try to find the main content area
    main = soup.find("main")
    if not main:
        main = soup.find("div", {"id": "main-content"})
    if not main:
        return ""
    # Remove navigation, sidebars, etc.
    for aside in main.find_all(["aside", "nav", "header", "footer"]):
        aside.decompose()
    return normalize_quotes(main.get_text("\n", strip=True))


def extract_question(soup):
    question = soup.find("div", class_="topic-identifier font-16 font-md-20")
    if question:
        return normalize_quotes(question.get_text(strip=True))
    return ""


# Extract arguments (pro/con)
def extract_argument_sections(soup, section_class):
    arguments = []
    for section in soup.find_all("section", class_=section_class):
        title = section.find("h2")
        title_text = normalize_quotes(title.get_text(strip=True)) if title else ""

        # Get all content elements, not just paragraphs
        content_elements = section.find_all(
            ["p", "blockquote", "ul", "ol"], class_="topic-paragraph"
        )
        argument_text = []
        footnotes = set()

        for element in content_elements:
            # Handle different element types
            if element.name == "p":
                text = normalize_quotes(element.get_text(" ", strip=True))
                argument_text.append(text)
            elif element.name == "blockquote":
                # Get all paragraphs within blockquote
                quote_texts = [
                    normalize_quotes(p.get_text(" ", strip=True))
                    for p in element.find_all("p")
                ]
                if quote_texts:
                    argument_text.append("> " + "\n> ".join(quote_texts))
            elif element.name in ["ul", "ol"]:
                # Handle lists
                list_items = []
                for li in element.find_all("li"):
                    item_text = normalize_quotes(li.get_text(" ", strip=True))
                    list_items.append(f"- {item_text}")
                if list_items:
                    argument_text.append("\n".join(list_items))

            # Extract footnotes from any element type
            for a in element.find_all("a", href=True):
                match = re.search(r"#pcref-\d+-([\d]+)", a["href"])
                if match:
                    footnotes.add(match.group(1))

        arguments.append(
            {
                "title": title_text,
                "text": "\n\n".join(
                    argument_text
                ),  # Use double newline for better readability
                "footnotes": sorted(footnotes),
            }
        )
    return arguments


# Extract quotes (pro/con)
def extract_quotes_section(soup, section_type, base_url=None, deep_search=True):
    # print(f"[DEBUG] Extracting {section_type} quotes...")
    # section_type: 'pro' or 'con'
    def blockquotes_with_footnotes(start_header, section_type):
        quotes = []
        sib = start_header.find_next_sibling()
        count = 0
        while sib:
            # Direct blockquote sibling
            if sib.name == "blockquote":
                text = "\n".join(
                    normalize_quotes(p.get_text(" ", strip=True))
                    for p in sib.find_all(["p", "ul", "ol"])
                )
                context = ""
                prev = sib.find_previous_sibling()
                if prev and prev.name in ["p", "h2", "h3"]:
                    context = normalize_quotes(prev.get_text(" ", strip=True))
                footnotes = set()
                for a in sib.find_all("a", href=True):
                    match = re.search(r"#pcref-\d+-([\d]+)", a["href"])
                    if match:
                        footnotes.add(match.group(1))
                # print(f"[DEBUG] Sibling blockquote: context='{context}', text='{text}', footnotes={footnotes}")
                quote_text = text
                if context:
                    quote_text = f"{context}\n\n{text}"
                quotes.append({"text": quote_text, "footnotes": sorted(footnotes)})
                # print(f"[DEBUG] Added quote: {quote_text} | footnotes: {sorted(footnotes)}")
                count += 1
            # Section sibling containing a blockquote
            elif sib.name == "section":
                blockquote = sib.find("blockquote")
                if blockquote:
                    text = "\n".join(
                        normalize_quotes(p.get_text(" ", strip=True))
                        for p in blockquote.find_all(["p", "ul", "ol"])
                    )
                    context = ""
                    prev = blockquote.find_previous_sibling()
                    if prev and prev.name in ["p", "h2", "h3"]:
                        context = normalize_quotes(prev.get_text(" ", strip=True))
                    footnotes = set()
                    for a in blockquote.find_all("a", href=True):
                        match = re.search(r"#pcref-\d+-([\d]+)", a["href"])
                        if match:
                            footnotes.add(match.group(1))
                    # print(f"[DEBUG] Section blockquote: context='{context}', text='{text}', footnotes={footnotes}")
                    quote_text = text
                    if context:
                        quote_text = f"{context}\n\n{text}"
                    quotes.append({"text": quote_text, "footnotes": sorted(footnotes)})
                    # print(f"[DEBUG] Added quote: {quote_text} | footnotes: {sorted(footnotes)}")
                    count += 1
            if sib.name in ["h1", "h2", "h3"]:
                break
            sib = sib.find_next_sibling()
        # Deep search: also look for all descendant blockquotes in the parent <section>
        if (
            deep_search
            and hasattr(start_header, "parent")
            and start_header.parent.name == "section"
        ):
            parent_section = start_header.parent
            for blockquote in parent_section.find_all("blockquote"):
                text = "\n".join(
                    normalize_quotes(p.get_text(" ", strip=True))
                    for p in blockquote.find_all(["p", "ul", "ol"])
                )
                context = ""
                prev = blockquote.find_previous_sibling()
                if prev and prev.name in ["p", "h2", "h3"]:
                    context = normalize_quotes(prev.get_text(" ", strip=True))
                footnotes = set()
                for a in blockquote.find_all("a", href=True):
                    match = re.search(r"#pcref-\d+-([\d]+)", a["href"])
                    if match:
                        footnotes.add(match.group(1))
                # print(f"[DEBUG] Deep search blockquote: context='{context}', text='{text}', footnotes={footnotes}")
                quote_text = text
                if context:
                    quote_text = f"{context}\n\n{text}"
                # Avoid duplicates
                if quote_text not in [q["text"] for q in quotes]:
                    quotes.append({"text": quote_text, "footnotes": sorted(footnotes)})
                    # print(f"[DEBUG] Added quote (deep): {quote_text} | footnotes: {sorted(footnotes)}")

        # Add section headers to quote texts
        for i, quote in enumerate(quotes, 1):
            quote["text"] = f"\n### {section_type.title()} {i}:\n{quote['text']}"
        return quotes

    # 1. Try to find the header (h1/h2/h3) called 'pro quotes' or 'con quotes'
    header = None
    for tag in ["h1", "h2", "h3"]:
        regex = re.compile(f"{section_type} quotes", re.IGNORECASE)
        header = soup.find(tag, string=regex)
        if header:
            break
    if header:
        quotes = blockquotes_with_footnotes(header, section_type)
        return quotes

    # 2. If no such header, look for an <a> tag with the same text, and if found, load the linked page and repeat the search
    regex = re.compile(f"{section_type} quotes", re.IGNORECASE)
    link = soup.find("a", string=regex)
    if link and link.has_attr("href") and not link["href"].startswith("#") and base_url:
        quotes_url = urljoin(base_url, link["href"])
        resp = requests.get(quotes_url)
        if resp.status_code == 200:
            quotes_soup = BeautifulSoup(resp.text, "html.parser")
            found_header = False
            for tag in ["h1", "h2", "h3"]:
                header = quotes_soup.find(tag, string=regex)
                if header:
                    quotes = blockquotes_with_footnotes(header, section_type)
                    return quotes
                    found_header = True
            # Fallback: if no header but blockquotes exist, extract all blockquotes with footnotes
            blockquotes = quotes_soup.find_all("blockquote")
            quotes = []
            for bq in blockquotes:
                text = "\n".join(
                    normalize_quotes(p.get_text(" ", strip=True))
                    for p in bq.find_all(["p", "ul", "ol"])
                )
                context = ""
                prev = bq.find_previous_sibling()
                if prev and prev.name in ["p", "h2", "h3"]:
                    context = normalize_quotes(prev.get_text(" ", strip=True))
                footnotes = set()
                for a in bq.find_all("a", href=True):
                    match = re.search(r"#pcref-\d+-([\d]+)", a["href"])
                    if match:
                        footnotes.add(match.group(1))
                # print(f"[DEBUG] Fallback blockquote: context='{context}', text='{text}', footnotes={footnotes}")
                quote_text = text
                if context:
                    quote_text = f"{context}\n\n{text}"
                quotes.append({"text": quote_text, "footnotes": sorted(footnotes)})
                # print(f"[DEBUG] Added quote (fallback): {quote_text} | footnotes: {sorted(footnotes)}")
            for i, quote in enumerate(quotes, 1):
                quote["text"] = f"\n### {section_type.title()} {i}:\n{quote['text']}"
            return quotes
    return []


def find_sources_section(soup):
    # Method 1: Look for h2 with class h1 containing "Sources"
    sources_header = soup.find(
        lambda tag: tag.name == "h2"
        and tag.get_text(strip=True) == "Sources"
        and tag.get("class")
        and "h1" in tag.get("class")
    )
    # Method 2: Look for any heading containing "Sources"
    if not sources_header:
        sources_header = soup.find(
            lambda tag: tag.name in ["h1", "h2", "h3"]
            and "Sources" in tag.get_text(strip=True)
        )
    # Method 3: Look for a section with id containing "sources" or "references"
    if not sources_header:
        sources_section = soup.find(
            lambda tag: tag.name == "section"
            and tag.get("id")
            and (
                "sources" in tag.get("id").lower()
                or "references" in tag.get("id").lower()
            )
        )
        if sources_section:
            sources_header = sources_section.find(["h1", "h2", "h3"])
    # Method 4: Look for a <section> with class 'sources' and an <h2> with text 'Sources'
    if not sources_header:
        sources_section = soup.find("section", class_="sources")
        if sources_section:
            h2 = sources_section.find(
                "h2", string=lambda s: s and s.strip() == "Sources"
            )
            if h2:
                sources_header = h2
    return sources_header


def extract_footnotes(soup, url):
    footnotes = {}
    sources_header = find_sources_section(soup)
    if sources_header:
        next_ol = sources_header.find_next("ol")
        if not next_ol:
            parent_section = sources_header.find_parent(["section", "div"])
            if parent_section and "sources" in (parent_section.get("class") or []):
                next_ol = parent_section.find("ol")
        if next_ol:
            for li in next_ol.find_all("li"):
                num = None
                if li.has_attr("id"):
                    m = re.search(r"-(\d+)$", li["id"])
                    if m:
                        num = m.group(1)
                if num:
                    citation = normalize_quotes(li.get_text(" ", strip=True))
                    link = None
                    a = li.find("a", href=True)
                    if a:
                        link = a["href"]
                    footnotes[num] = {"citation": citation, "link": link}
    if not footnotes:
        for ol in soup.find_all("ol"):
            for li in ol.find_all("li"):
                num = None
                if li.has_attr("id"):
                    m = re.search(r"-(\d+)$", li["id"])
                    if m:
                        num = m.group(1)
                if not num:
                    m = re.match(r"\[(\d+)\]", li.get_text())
                    if m:
                        num = m.group(1)
                if num:
                    citation = normalize_quotes(li.get_text(" ", strip=True))
                    link = None
                    a = li.find("a", href=True)
                    if a:
                        link = a["href"]
                    footnotes[num] = {"citation": citation, "link": link}
    # Try to find a 'Sources' link in the table of contents and follow it
    if not footnotes and url:
        sources_link = None
        for nav in soup.find_all(["nav", "ul"]):
            link = nav.find("a", string=re.compile(r"^sources$", re.IGNORECASE))
            if link and link.has_attr("href"):
                sources_link = link["href"]
                break
        if sources_link:
            sources_url = urljoin(url, sources_link)
            print(f"Following sources link from TOC: {sources_url}")
            resp = requests.get(sources_url)
            if resp.status_code == 200:
                sources_soup = BeautifulSoup(resp.text, "html.parser")
                sources_header = find_sources_section(sources_soup)
                if sources_header:
                    next_ol = sources_header.find_next("ol")
                    if not next_ol:
                        parent_section = sources_header.find_parent(["section", "div"])
                        if parent_section and "sources" in (
                            parent_section.get("class") or []
                        ):
                            next_ol = parent_section.find("ol")
                    if next_ol:
                        for li in next_ol.find_all("li"):
                            num = None
                            if li.has_attr("id"):
                                m = re.search(r"-(\d+)$", li["id"])
                                if m:
                                    num = m.group(1)
                            if num:
                                citation = normalize_quotes(
                                    li.get_text(" ", strip=True)
                                )
                                link = None
                                a = li.find("a", href=True)
                                if a:
                                    link = a["href"]
                                footnotes[num] = {"citation": citation, "link": link}
    # If still no sources, try fallback suffixes
    if not footnotes and url:
        if not url.endswith("/"):
            base_url = url + "/"
        else:
            base_url = url
        found_sources = False
        tried_urls = []
        suffixes = [
            "Assessment-Quiz",
            "assessment-quiz",
            "1-minute-Survey",
            "1-Minute-Survey",
            "Cons",
            "Cons-Quotes",
            "Con-Quotes",
            "Pros",
            "Pros-Quotes",
            "Pro-Quotes",
            "Sources",
            "sources",
        ]
        for suffix in suffixes:
            quiz_url = urljoin(base_url, suffix)
            tried_urls.append(quiz_url)
            resp = requests.get(quiz_url)
            if resp.status_code == 200:
                found_sources = True
                break
        if found_sources:
            quiz_soup = BeautifulSoup(resp.text, "html.parser")
            sources_header = find_sources_section(quiz_soup)
            if sources_header:
                next_ol = sources_header.find_next("ol")
                if not next_ol:
                    parent_section = sources_header.find_parent(["section", "div"])
                    if parent_section and "sources" in (
                        parent_section.get("class") or []
                    ):
                        next_ol = parent_section.find("ol")
                if next_ol:
                    for li in next_ol.find_all("li"):
                        num = None
                        if li.has_attr("id"):
                            m = re.search(r"-(\d+)$", li["id"])
                            if m:
                                num = m.group(1)
                        if num:
                            citation = normalize_quotes(li.get_text(" ", strip=True))
                            link = None
                            a = li.find("a", href=True)
                            if a:
                                link = a["href"]
                            footnotes[num] = {"citation": citation, "link": link}
        else:
            print(
                f"Warning: No sources found for {url} after trying all suffixes. Tried: {tried_urls}"
            )
    return footnotes


def write_argument_file(path, arg, footnotes):
    with open(path, "w") as f:
        if "title" in arg and arg["title"]:
            f.write(f'### {arg["title"]}\n\n')
        f.write(f'{arg["text"]}\n\n')
        if arg["footnotes"]:
            f.write("---\n")
            for n in arg["footnotes"]:
                if n in footnotes:
                    info = footnotes[n]
                    link = f' ([link]({info["link"]}))' if info["link"] else ""
                    f.write(f'[{n}]: {info["citation"]}{link}\n')


def write_quotes_individual_files(quotes, footnotes, out_dir):
    for i, quote in enumerate(quotes, 1):
        path = os.path.join(out_dir, f"quote{i}.md")
        with open(path, "w") as f:
            if quote["text"]:
                f.write(f'{quote["text"]}\n\n')
                if quote["footnotes"]:
                    f.write("---\n")
                    for n in quote["footnotes"]:
                        if n in footnotes:
                            info = footnotes[n]
                            link = f' ([link]({info["link"]}))' if info["link"] else ""
                            f.write(f'[{n}]: {info["citation"]}{link}\n')


def write_sources_file(path, footnotes):
    with open(path, "w") as f:
        for i, (num, info) in enumerate(
            sorted(footnotes.items(), key=lambda x: int(x[0])), 1
        ):
            link = f' ([link]({info["link"]}))' if info["link"] else ""
            f.write(f'[{i}] {info["citation"]}{link}\n')


def save_full_text(
    path, main_content, pro_args, pro_quotes, con_args, con_quotes, footnotes
):
    with open(path, "w") as f:
        f.write("## Main Content\n\n")
        f.write(main_content + "\n\n")
        for label, items in [
            ("Pro Arguments", pro_args),
            ("Pro Quotes", pro_quotes),
            ("Con Arguments", con_args),
            ("Con Quotes", con_quotes),
        ]:
            if items:
                f.write(f"## {label}\n\n")
                for i, arg in enumerate(items, 1):
                    if "title" in arg and arg["title"]:
                        f.write(f'### {arg["title"]}\n')
                    f.write(f'{arg["text"]}\n')
                    if arg["footnotes"]:
                        f.write("---\n")
                        for n in arg["footnotes"]:
                            if n in footnotes:
                                info = footnotes[n]
                                link = (
                                    f' ([link]({info["link"]}))' if info["link"] else ""
                                )
                                f.write(f'[{n}]: {info["citation"]}{link}\n')
                    f.write("\n")
        if footnotes:
            f.write("## Footnotes\n\n")
            for num, info in footnotes.items():
                link = f' ([link]({info["link"]}))' if info["link"] else ""
                f.write(f'[{num}]: {info["citation"]}{link}\n')


def get_toc_argument_links(soup, base_url):
    pro_link = None
    con_link = None
    for nav in soup.find_all(["nav", "ul"]):
        for link in nav.find_all("a", string=True):
            text = link.get_text(strip=True)
            # Match 'Pro', 'Pros', 'Con', or 'Cons' as a whole word, but not 'Pro Quotes' or 'Con Quotes'
            if re.fullmatch(r"pros?", text, re.IGNORECASE):
                pro_link = urljoin(base_url, link["href"])
            elif re.fullmatch(r"cons?", text, re.IGNORECASE):
                con_link = urljoin(base_url, link["href"])
    return pro_link, con_link


# Main scraping function


def scrape_debate(url):
    sleep(0.5 + random.random())
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    slug = slugify(url)
    question = extract_question(soup)

    # Try to get pro/con argument links from TOC
    pro_link, con_link = get_toc_argument_links(soup, url)
    pro_args = []
    con_args = []
    if pro_link:
        print(f"Following TOC pro link: {pro_link}")
        resp_pro = requests.get(pro_link)
        if resp_pro.status_code == 200:
            soup_pro = BeautifulSoup(resp_pro.text, "html.parser")
            pro_args = extract_argument_sections(soup_pro, "pro")
            print(f"Found {len(pro_args)} pro arguments from TOC link.")
    if con_link:
        print(f"Following TOC con link: {con_link}")
        resp_con = requests.get(con_link)
        if resp_con.status_code == 200:
            soup_con = BeautifulSoup(resp_con.text, "html.parser")
            con_args = extract_argument_sections(soup_con, "con")
            print(f"Found {len(con_args)} con arguments from TOC link.")
    # Fallback to extracting from main page if not found via TOC
    if not pro_args:
        pro_args = extract_argument_sections(soup, "pro")
        print(f"Found {len(pro_args)} pro arguments from main page.")
    if not con_args:
        con_args = extract_argument_sections(soup, "con")
        print(f"Found {len(con_args)} con arguments from main page.")

    pro_quotes = extract_quotes_section(soup, "pro", base_url=url)
    con_quotes = extract_quotes_section(soup, "con", base_url=url)
    footnotes = extract_footnotes(soup, url)

    title_regex = re.compile(r"(?:\s*#+)?\s*(Pro|Con)\s+\d+:\s+", re.I)
    for arg in pro_args + con_args + pro_quotes + con_quotes:
        arg["text"] = re.sub(title_regex, "", arg["text"])

    # Prepare pros and cons as lists of strings (combine title and text)
    pros = []
    for arg in pro_args + pro_quotes:
        pros.append(arg["text"])
    cons = []
    for arg in con_args + con_quotes:
        cons.append(arg["text"])

    # Prepare sources as list of dicts
    sources = {}
    for num, info in sorted(footnotes.items(), key=lambda x: int(x[0])):
        sources[f"[{num}]"] = info["citation"]

    # Write JSON file
    out_path = os.path.join(OUTPUT_DIR, f"{slug}.json")
    with open(out_path, "w") as f:
        json.dump(
            {"question": question, "pros": pros, "cons": cons, "sources": sources},
            f,
            indent=2,
        )

    return pro_args, con_args, pro_quotes, con_quotes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Sample 5 topics")
    args = parser.parse_args()
    num_topics = 0
    num_pro_arguments = 0
    num_con_arguments = 0
    num_pro_quotes = 0
    num_con_quotes = 0
    ensure_dir(OUTPUT_DIR)
    with open(PROCON_LINKS_FILE) as f:
        urls = [line.strip() for line in f if line.strip()]
    if args.sample:
        urls = urls[:5]
    for url in urls:
        print(f"Scraping {url} ...")
        try:
            pro_args, con_args, pro_quotes, con_quotes = scrape_debate(url)
            num_topics += 1
            num_pro_arguments += len(pro_args) if pro_args else 0
            num_con_arguments += len(con_args) if con_args else 0
            num_pro_quotes += len(pro_quotes) if pro_quotes else 0
            num_con_quotes += len(con_quotes) if con_quotes else 0
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    print(f"Number of topics: {num_topics}")
    print(f"Number of pro arguments: {num_pro_arguments}")
    print(f"Number of con arguments: {num_con_arguments}")
    print(f"Number of pro quotes: {num_pro_quotes}")
    print(f"Number of con quotes: {num_con_quotes}")
