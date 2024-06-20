import requests, shutil, os, rich, time, re

from tqdm.auto import tqdm
from PyPDF2 import PdfReader
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

section = "CV"  # LG(machine learning), CL(computation and language)
pdf_folder = "arxiv_cs_{section}"
os.mkdir(pdf_folder)

arxiv_url = "http://arxiv.org/"
arxiv_tree_url = f"http://arxiv.org/list/cs.{section}"

# initialize driver
drive_opts = Options()
drive_opts.add_argument("-headless")
driver = webdriver.Firefox(options=drive_opts)
print("Driver init")

# load page content
page = driver.get(f"{arxiv_tree_url}/recent?skip=0&show=2000")
print("Driver loaded...🔥")

# get all sections in computer science
cs_sections = driver.find_elements(By.ID, "cs")
section_acronyms = [link.get_attribute("id") for link in cs_sections]
section_acronyms = [sect.upper() for sect in section_acronyms]  # type: ignore


# get all pdf links and titles
section_links = driver.find_elements(By.ID, f"cs.{section}")
titles = driver.find_elements(By.CLASS_NAME, "list-title.mathjax")
print(f"Number of links  => {len(section_links)}")


pdf_links = [link.get_attribute("src") for link in section_links]
paper_titles = [tit.text for tit in titles]
print(len(pdf_links), len(paper_titles))
print(f"Link and titles retrieved. e.g => {paper_titles[0]: pdf_links[0]}")

# quit driver instance
driver.quit()


# download files wuth corresponding paper titles
def download_files(link_list: list, title_list: list):
    k = 0

    for file_link, title in tqdm(zip(link_list, title_list)):
        try:
            response = requests.get(str(file_link), stream=True)
            file_path = os.path.join(pdf_folder, f"{title}_{k}.pdf")

            with open(file_path, "wb") as file:
                file.write(response.content)
                response.raw.decode_content = True

                shutil.copyfileobj(response.raw, file)
                # print("File downloaded successfully")
            k += 1

        except Exception as e:
            rich.print(f"[red]{e}")
            continue

    print(f"{k} pdfs downloaded from arxiv 🍻")


download_files(pdf_links, paper_titles)


####
# for merging all the pdfs into a single text file

print("Begin pdf merging")


def clean_text(text):  # regex cleaning function
    regex_format = (
        r"[^\w\s\d]"  # Leaves behind only text, digits, and whitespace characters
    )
    cleaned_text = re.sub(regex_format, "", text)

    return cleaned_text


def pdf2text_file(pdf_file: str):  # extrcat text from pdf
    extract_text = ""
    pdf_read = PdfReader(pdf_file)

    for page_num in tqdm(range(len(pdf_read.pages))):
        page = pdf_read.pages[page_num]
        extract_text += page.extract_text()
        extract_text = clean_text(extract_text)

    return extract_text


def merge_pdfs(file_list: list, out_text_file: str):
    text_corpus = ""
    try:
        for pdf in tqdm(file_list, colour="blue"):  # extract from pdf in queue
            try:
                text_extract = pdf2text_file(pdf)
                text_corpus += text_extract  # concat text and add space before the next one is added
                text_corpus += "\n  "
                rich.print(f"Extracted [white] {pdf}")

                with open(out_text_file, "w", encoding="utf-8") as file:
                    file.write(text_corpus)  # write to text file

            except Exception as e:
                print(e)

                continue

        # success message
        rich.print(
            f"{len(pdf_list)} research paper PDFs extracted to single text file [bold green]{out_text_file}[/bold green] of size {shutil.disk_usage(out_text_file)}"
        )

        return text_corpus
    # exception hanndling and colored output
    except Exception as e:
        rich.print(f"[bold red] Error in extraction --> {e}")



 
def arxiv_scraper(section: str):
    arxiv_tree_url = f"http://arxiv.org/list/cs.{section}"

    section = "PL"  # LG(machine learning), CL(computation and language)
    pdf_folder = f"arxiv_cs_{section}"
    os.mkdir(pdf_folder)

    # initialize driver
    drive_opts = Options()
    drive_opts.add_argument("-headless")
    driver = webdriver.Firefox(options=drive_opts)
    print("Driver init")

    # load page content
    driver.get(f"{arxiv_tree_url}/recent?skip=0&show=2000")
    print("Driver loaded...🔥")

    # get all pdf links and titles
    pdf_links = driver.find_elements(By.LINK_TEXT, 'pdf')
    titles = driver.find_elements(By.CLASS_NAME, "list-title")
    print(f"Number of links  => {len(pdf_links)}")
    print(f"Number of titles  => {len(titles)}")

    paper_titles = [tit.text for tit in titles]
    print(len(pdf_links), len(paper_titles))

    pdf_links = [link.get_attribute("href") for link in pdf_links]
    print(f"Link and titles retrieved. e.g => {paper_titles[0]}: {pdf_links[0]}")

    driver.quit()
    print('driver terminated')

    print("Begin pdf merging")
    pdf_list = os.listdir(pdf_folder)
    pdf_list = [os.path.join(os.getcwd(), pdf_folder, pdfile) for pdfile in pdf_list]

    out_text_file = f"arxiv_cspapers_{section}.txt"
    merge_pdfs(pdf_list, out_text_file)
    print("Extraction and merging complete ⚡️⚡️")
