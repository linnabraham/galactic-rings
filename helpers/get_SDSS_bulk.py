#!/bin/env python
import sys
# sys.path.append("get_images")
# import ps1image as ps
import argparse, os
import logging
import concurrent.futures
import random
import time
import requests

import urllib
from astropy import units as u

def get_SDSS_url(ra, dec, impix=256, imsize=1*u.arcmin):

    cutoutbaseurl = 'http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'
    query_string = urllib.parse.urlencode(dict(ra=ra,dec=dec,width=impix, height=impix, scale=imsize.to(u.arcsec).value/impix))
    url = cutoutbaseurl + '?' + query_string
    return url

# a custom generator to create list of lists from pandas dataframe
def chunks(lst, n):
    """yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def fetch_url(entry):
    downloaded = False
    tries = 5
    count = 0
    uri, path = entry
    #set a default timeout value
    timeout = gtimeout

    randSleep = random.randint(1, 101) / 20.

    time.sleep(randSleep)

    while not downloaded and count < tries:
        try:
            r = requests.get(uri, stream=True, timeout=timeout)
        except Exception as e:
            print(e)
            count += 1
            timeout *= 2
            llog.warning(f"timeout on attempt number {count}/{tries}. Increasing to {timeout}s")
            continue

        if r.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in r:
                    f.write(chunk)
            return path
        else:
            count += 1
            llog.warning(f"Getting status code {r.status_code} on download attempt {count}/{tries}.")
            downloaded = False

    return None

def multiobject_download(
    urlList,
    downloadDirectory,
    log,
    filenames,
    timeStamp=True,
    timeout=180,
    concurrentDownloads=10

):
    import sys
    import os
    import re

    # TIMEOUT IN SECONDS
    global gtimeout
    global llog
    llog = log
    gtimeout = float(timeout)

    # BUILD THE 2D ARRAY FOR MULTI_THREADED DOWNLOADS
    thisArray = []

    totalCount = len(urlList)
    # IF ONLY ONE DOWNLOAD DIRECORY
    if not isinstance(downloadDirectory, list):
        for i, url in enumerate(urlList):
            filename=filenames[i]
            # GENERATE THE LOCAL FILE URL
            localFilepath = downloadDirectory + "/" + filename
            thisArray.extend([[url, localFilepath]])

    # CONCURRENTLY DOWNLOAD URLS
    # print(thisArray,"This array")
    # results = ThreadPool(concurrentDownloads).imap_unordered(
        # fetch_url, thisArray)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(fetch_url,thisArray)
    urlNum = 0
    returnPaths = []
    for path in results:
        returnPaths.append(path)
        urlNum += 1
        if urlNum > 1:
            # CURSOR UP ONE LINE AND CLEAR LINE
            sys.stdout.write("\x1b[1A\x1b[2K")
        percent = (float(urlNum) / float(totalCount)) * 100.
        print("  %(urlNum)s / %(totalCount)s (%(percent)1.1f%%) URLs downloaded" % locals())

    localPaths = []
    localPaths[:] = [o[1] for o in thisArray if o[1] in returnPaths]
    return localPaths


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('list', help="path of csv file containing RA,DEC values")
    parser.add_argument('-o', default="images", help="Output directory for storing images")
    args = parser.parse_args()


    download_list = args.o+"/downloaded.csv"
    url_notfound = args.o+"/urlnotfound.csv"
    downdir = args.o+"/panstamps"


    if not os.path.exists(args.o):
        print(f"Directory {args.o} doesn't exist. Creating...")
        os.makedirs(args.o)
    if not os.path.exists(downdir):
        os.makedirs(downdir)


    print(f"Download list is being appended to {download_list}")

    # intiate a logger object
    logger = logging.getLogger("my logger")


    # read catalogue file into dataframe
    import pandas as pd
    catalogue_file = args.list


    # catalogue = pd.read_csv(catalogue_file,dtype={"dr7objid":int})
    data = pd.read_csv(catalogue_file)
    print("Select RA and DEC columns:",[column for column in data.columns])
    racol = input("RA column:")
    deccol = input("DEC column:")
    objidcol = input("ObjID column:")



    # retrieved_ids is non empty only if downloaded_list is present
    retrieved_ids = []

    try:
        import os
        downloaded=pd.read_csv(download_list,names=['paths'],dtype={"paths":str})
        print("[INFO]: Resuming previous download job")
    except:
        print("[INFO]: This is a fresh download...")
    else:
        # fix issue with pandas reading some values as floats
        # downloaded["paths"] = downloaded["paths"].astype('|S')
        paths=list(downloaded["paths"])
        if not any(isinstance(item, str) for item in paths):
            print ("Found non-strings in the list of paths")

        retrieved_ids=[os.path.basename(x).rstrip(".jpeg") for x in paths if isinstance(x,str)]
        print(f"length of retrieved ids {len(retrieved_ids)}")
        # allUrls = [ x for x in objids if x not in retrieved_ids ]


    # first create a multi-dimensional list from dataframe using df.itertuples and pass this to the chunks function
    batches = list(chunks(list(map(list,data[[racol,deccol,objidcol]].itertuples(index=False))),10))

    for batch in batches:
        concurrentDownloads = len(batch)
        # store the object ids in each batch to write to log if download is successful
        objids=[]
        # a list to store the image urls for all sources in batch for parallel downloads
        allUrls=[]
        for row in batch:
            RA=row[0]
            DEC=row[1]
            # oid=str(RA)+"+"+str(DEC)
            # oid="{ra:.4f}+{dec:.4f}".format(ra=RA,dec=DEC)
            oid=str(row[2])
            print(oid)


            if str(oid) in retrieved_ids:
                print("skipped")
                continue
            objids.append(oid+".jpeg")

            import csv
            try:
                # url = ps.geturl(ra=RA,dec=DEC,color=True)
                url = get_SDSS_url(ra=RA, dec=DEC)
            except:
                with open(url_notfound,"a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([f"{RA},{DEC} not found"])
            else:
                allUrls.append(url)

        # Do the actual parallelized downloads
        st=time.time()
        print("len of urls ,",len(allUrls),len(objids))
        localUrls = multiobject_download(
            urlList=allUrls,
            downloadDirectory=downdir,
            log=logger,
            timeout=180,
            concurrentDownloads=concurrentDownloads,
            filenames=objids
        )

        t = time.time()-st
        print(f"Time taken:{t}s")
        import csv
        with open(download_list,"a") as csvfile:
            writer = csv.writer(csvfile)
            for item in localUrls:
                writer.writerow([item])
