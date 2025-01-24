########################
# Script for downloading images from Reddit, posts were scraped from: https://arctic-shift.photon-reddit.com/download-tool

from typing import BinaryIO, Iterator, Iterable
import pandas as pd
import traceback
import zstandard
import datetime
import requests
import time
import tqdm
import sys
import os
import re

subreddit = "dndmaps"
fileOrFolderPath = f'../data/posts/r_{subreddit}_posts.jsonl'
output_dir = f"../data/imgs/{subreddit}"

# Imgs downloading config:
download_images = False
compress_images = False
compress_quality = 70

# Make dir if not exists
os.makedirs(output_dir, exist_ok=True)

# Add captions to imgs
add_captions = True

def formatTime(seconds: float) -> str:
	if seconds == 0:
		return "0s"
	if seconds < 0.001:
		return f"{seconds * 1_000_000:.1f}µs"
	if seconds < 1:
		return f"{seconds * 1_000:.2f}ms"
	elapsedHr = int(seconds // 3600)
	elapsedMin = int((seconds % 3600) // 60)
	elapsedSec = int(seconds % 60)
	return f"{elapsedHr:02}:{elapsedMin:02}:{elapsedSec:02}"

class FileProgressLog:
	file: BinaryIO
	fileSize: int
	i: int
	startTime: float
	printEvery: int
	maxLineLength: int

	def __init__(self, path: str, file: BinaryIO):
		self.file = file
		self.fileSize = os.path.getsize(path)
		self.i = 0
		self.startTime = time.time()
		self.printEvery = 10_000
		self.maxLineLength = 0
	
	def onRow(self):
		self.i += 1
		if self.i % self.printEvery == 0 and self.i > 0:
			self.logProgress()
		
	def logProgress(self, end=""):
		progress = self.file.tell() / self.fileSize if not self.file.closed else 1
		elapsed = time.time() - self.startTime
		remaining = (elapsed / progress - elapsed) if progress > 0 else 0
		timePerRow = elapsed / self.i
		printStr = f"{self.i:,} - {progress:.2%} - elapsed: {formatTime(elapsed)} - remaining: {formatTime(remaining)} - {formatTime(timePerRow)}/row"
		self.maxLineLength = max(self.maxLineLength, len(printStr))
		printStr = printStr.ljust(self.maxLineLength)
		print(f"\r{printStr}", end=end)

		if timePerRow < 20/1000/1000:
			self.printEvery = 20_000
		elif timePerRow < 50/1000/1000:
			self.printEvery = 10_000
		else:
			self.printEvery = 5_000


try:
	import orjson as json
except ImportError:
	import json
	print("Recommended to install 'orjson' for faster JSON parsing")

def getZstFileJsonStream(f: BinaryIO, chunk_size=1024*1024*10) -> Iterator[dict]:
	decompressor = zstandard.ZstdDecompressor(max_window_size=2**31)
	currentString = ""
	def yieldLinesJson():
		nonlocal currentString
		lines = currentString.split("\n")
		currentString = lines[-1]
		for line in lines[:-1]:
			try:
				yield json.loads(line)
			except json.JSONDecodeError:
				print("Error parsing line: " + line)
				traceback.print_exc()
				continue
	zstReader = decompressor.stream_reader(f)
	while True:
		try:
			chunk = zstReader.read(chunk_size)
		except zstandard.ZstdError:
			print("Error reading zst chunk")
			traceback.print_exc()
			break
		if not chunk:
			break
		currentString += chunk.decode("utf-8", "replace")
		
		yield from yieldLinesJson()
	
	yield from yieldLinesJson()
	
	if len(currentString) > 0:
		try:
			yield json.loads(currentString)
		except json.JSONDecodeError:
			print("Error parsing line: " + currentString)
			print(traceback.format_exc())
			pass

def getJsonLinesFileJsonStream(f: BinaryIO) -> Iterator[dict]:
	for line in f:
		line = line.decode("utf-8", errors="replace")
		try:
			yield json.loads(line)
		except json.JSONDecodeError:
			print("Error parsing line: " + line)
			traceback.print_exc()
			continue

def getFileJsonStream(path: str, f: BinaryIO) -> Iterator[dict]|None:
	if path.endswith(".jsonl"):
		return getJsonLinesFileJsonStream(f)
	elif path.endswith(".zst"):
		return getZstFileJsonStream(f)
	else:
		return None


version = sys.version_info
if version.major < 3 or (version.major == 3 and version.minor < 10):
	raise RuntimeError("This script requires Python 3.10 or higher")
import os





recursive = False


def processFile(path: str):
	print(f"Processing file {path}")
	post_data = []
	with open(path, "rb") as f:
		jsonStream = getFileJsonStream(path, f)
		if jsonStream is None:
			print(f"Skipping unknown file {path}")
			return
		progressLog = FileProgressLog(path, f)
		for row in jsonStream:
			progressLog.onRow()
			
			# Permalink, Id, Subreddit, User, Type, Title, Content, Timestamp, NoLikes, NoReplies, ImagesUrls
			
			permalink = row["permalink"]
			id = row["id"]
			subreddit = row["subreddit"]
			user = row["author"]
			title = row["title"]
			content = row["selftext"]
			timestamp = datetime.datetime.fromtimestamp(row.get("created_utc", 0)).strftime('%Y-%m-%d %H:%M:%S')
			score = row["score"]
			replies = row["num_comments"]
			
			images_urls = []
			media_metadata = row.get('media_metadata')
			if isinstance(media_metadata, dict):
				for img in media_metadata.values():
					if isinstance(img, dict):
						if img.get('e') == 'Image':
							s = img.get('s')
							if isinstance(s, dict):
								url = s.get('u')
								if url:
									images_urls.append(url.replace("&amp;", "&"))
						else:
							pass
			elif isinstance(row.get('preview'), dict):
				images = row['preview'].get('images', [])
				for image in images:
					source = image.get('source', {})
					url = source.get('url')
					if url:
						images_urls.append(url.replace("&amp;", "&"))
			elif 'url' in row and row.get('post_hint') == 'image':
				images_urls = [row['url']]

			if not images_urls:
				continue

			post_data.append({
				"Permalink": permalink,
				"Id": id,
				"Subreddit": subreddit,
				"User": user,
				"Title": title,
				"Content": content,
				"Timestamp": timestamp,
				"NoLikes": score,
				"NoReplies": replies,
				"ImagesUrls": images_urls,
			})

			# print(f"Link: {permalink} - Id: {id} - r/: {subreddit} - User: {user} - Type: {type} - Title: {title} - Content: {content} - Time: {timestamp} - Score: {score} - Replies: {replies} - ImagesUrls: {images_urls}")

		progressLog.logProgress("\n")
		df = pd.DataFrame(post_data)
		return df
	

def processFolder(path: str):
	fileIterator: Iterable[str]
	if recursive:
		def recursiveFileIterator():
			for root, dirs, files in os.walk(path):
				for file in files:
					yield os.path.join(root, file)
		fileIterator = recursiveFileIterator()
	else:
		fileIterator = os.listdir(path)
		fileIterator = (os.path.join(path, file) for file in fileIterator)
	
	for i, file in enumerate(fileIterator):
		print(f"Processing file {i+1: 3} {file}")
		processFile(file)

if os.path.isdir(fileOrFolderPath):
	processFolder(fileOrFolderPath)
else:
	df = processFile(fileOrFolderPath)

print("Done :>")
df.head()

# save data to csv file   
df.to_csv(f'../data/posts/r_{subreddit}_posts.csv', index=False)

df['ImagesUrls'].head()

# Convert 'Timestamp' to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

posts_per_year = df.groupby(df['Timestamp'].dt.year).size()

print("Posts per Year:")
posts_per_year

filtered_df = df[df["NoReplies"] >= 10]
df = filtered_df.copy()

# ---
# 
# # Downloading
# 

import logging


successful_downloads = 0
failed_downloads = 0



def download_image(url, save_path, max_retries=2, timeout=20):
    global successful_downloads, failed_downloads
    attempt = 0
    while attempt < max_retries:
        try:
            logging.info(f"Attempt {attempt + 1} to download {url}")
            response = requests.get(url, stream=True, timeout=timeout)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                logging.info(f"Successfully downloaded {url} to {save_path}")
                successful_downloads += 1
                return True
            else:
                logging.warning(f"Failed to download {url} (status code: {response.status_code})")
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout occurred while downloading {url}.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request exception for {url}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error downloading {url}: {e}")
        
        attempt += 1
        if attempt < max_retries:
            logging.info(f"Retrying download for {url} (Attempt {attempt + 1}) after waiting for 5 seconds...")
            time.sleep(5)  # Wait before retrying
        else:
            logging.error(f"Skipping {url} after {max_retries} failed attempts.")
    failed_downloads += 1
    return False

if download_images:
    total_images = sum(len(row['ImagesUrls']) for idx, row in df.iterrows())
    with tqdm.tqdm(total=total_images, desc="Downloading images") as pbar:
        for idx, row in df.iterrows():
            
            post_id = row['Id']
            image_urls = row['ImagesUrls']
            
            for i, url in enumerate(image_urls):
                filename = f"{post_id}_row{idx}_img{i}.jpg"
                save_path = os.path.join(output_dir, filename)
                if not os.path.exists(save_path):
                    # Download the image
                    download_image(url, save_path)
                else:
                    logging.info(f"Skipping download for {url} as file already exists.")
                    successful_downloads += 1
                # Update the progress bar and display counts of downloads
                pbar.update(1)
                pbar.set_postfix({
                    "Successful": successful_downloads,
                    "Failed": failed_downloads,
                    "Remaining": total_images - (successful_downloads + failed_downloads)
                })

    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")

def generate_prompt_variations(title: str) -> list[str]:
    """Generate different prompt variations for a fantasy/DnD map title"""
    # Clean the title - remove common Reddit-specific patterns
    title = re.sub(r'\[.*?\]|\(.*?\)', '', title)  # Remove text in [] and ()
    title = re.sub(r'OC|WIP|4K|HD', '', title, flags=re.IGNORECASE)  # Remove common tags
    title = title.strip()
    
    # Base variations that describe the type of map
    variations = [
        f"fantasy map of {title}",
        f"DnD style map showing {title}",
        f"hand-drawn fantasy map depicting {title}",
        f"top-down view map of {title} in fantasy style",
    ]
    
    # Add variations based on common map styles
    if any(word in title.lower() for word in ['city', 'town', 'village']):
        variations.extend([
            f"medieval fantasy city map of {title}",
            f"detailed town layout of {title} in fantasy style",
            f"bird's eye view of fantasy settlement {title}"
        ])
    
    if any(word in title.lower() for word in ['dungeon', 'cave', 'lair']):
        variations.extend([
            f"fantasy dungeon map of {title}",
            f"DnD dungeon layout showing {title}",
            f"top-down dungeon design of {title}"
        ])
    
    if any(word in title.lower() for word in ['region', 'realm', 'kingdom', 'world']):
        variations.extend([
            f"fantasy world map of {title}",
            f"detailed fantasy region map showing {title}",
            f"hand-drawn fantasy realm of {title}"
        ])
    
    return variations

def sanitize_filename(text: str) -> str:
    """Convert text to a valid filename"""
    # Remove invalid filename characters
    valid_filename = re.sub(r'[<>:"/\\|?*]', '', text)
    # Replace spaces with underscores
    valid_filename = valid_filename.replace(' ', '_')
    # Limit length to avoid too long filenames
    if len(valid_filename) > 200:
        valid_filename = valid_filename[:200]
    return valid_filename

if add_captions:
    print("Adding captions to images...")

    # Create a mapping of old filenames to new filenames
    filename_mapping = {}

    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        post_id = row['Id']
        title = row['Title']
        
        # Generate prompt variations
        prompts = generate_prompt_variations(title)
        prompt_text = " || ".join(prompts)
        
        # For each image associated with this post
        for i in range(len(row['ImagesUrls'])):
            old_filename = f"{post_id}_row{idx}_img{i}.jpg"
            new_filename = f"{sanitize_filename(prompt_text)}__{post_id}_img{i}.jpg"
            
            old_path = os.path.join(output_dir, old_filename)
            new_path = os.path.join(output_dir, new_filename)
            
            if os.path.exists(old_path):
                filename_mapping[old_path] = new_path

    # Rename all files
    print(f"Renaming {len(filename_mapping)} files...")
    for old_path, new_path in tqdm.tqdm(filename_mapping.items()):
        try:
            os.rename(old_path, new_path)
        except Exception as e:
            print(f"Error renaming {old_path}: {e}")

    print("Done renaming files with captions!")