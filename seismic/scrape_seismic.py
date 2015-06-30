from datetime import datetime
import requests
import os

class CouldNotScrapeException(Exception):
    pass

import cPickle as pickle

def load_events(sta="mkar"):
    s = []
    for i in range(1, 100):
        try:
            with open("/home/dmoore/p_waves/%s_stuff_%d" % (sta, i * 1000), 'rb') as f:
                ss = pickle.load(f)
                s += ss
            print "loaded", i
        except IOError:
            with open("/home/dmoore/p_waves/%s_stuff_final" % (sta,), 'rb') as f:
                ss = pickle.load(f)
                s += ss
            print "loaded final"
            break

    return s



def extract_ev(page):
    if "No events were found" in page:
        raise CouldNotScrapeException()

    try:
        idx1 = page.index("<pre>")+6
        idx2 = page.index("STOP")
        csvpage = page[idx1:idx2]
        lines = csvpage.split("\n")


        prime_hcenter = -1
        hcenters = []
        for line in lines:
            if "PRIME" in line:
                break
            if not line.startswith("20"): continue
            lat = float(line[37:44])
            lon = float(line[46:54])
            try:
                smaj = float(line[56:60])
                smin = float(line[62:66])
                strike = int(line[68:70])
            except:
                smaj = 20.0
                smin = 20.0
                strike = 0
            depth = float(line[72:76])
            try:
                depth_err = float(line[79:82])
            except:
                depth_err = 0.05*depth + 1.0

            hcenters.append((lon, lat, smaj, smin, strike, depth, depth_err))
        if len(hcenters)==0:
            raise CouldNotScrapeException()
        else:
            return hcenters[prime_hcenter]
    except Exception as e:
        print e
        raise CouldNotScrapeException()

def scrape_isc(ev):
    lon = ev.lon
    lat = ev.lat

    sdt =  datetime.utcfromtimestamp(ev.time - 120)
    edt = datetime.utcfromtimestamp(ev.time + 120)

    stime = "%02d:%02d:%02d" % (sdt.hour, sdt.minute, sdt.second)
    etime = "%02d:%02d:%02d" % (edt.hour, edt.minute, edt.second)

    url = "http://isc-mirror.iris.washington.edu/cgi-bin/web-db-v4?out_format=ISF&request=COMPREHENSIVE&searchshape=CIRC&ctr_lat=%.2f&ctr_lon=%.2f&radius=80&max_dist_units=km&start_year=%d&start_month=%d&start_day=%d&start_time=%s&end_year=%d&end_month=%d&end_day=%d&end_time=%s&req_mag_agcy=Any" % (lat, lon, sdt.year, sdt.month, sdt.day, stime, edt.year, edt.month, edt.day, etime)


    #url = "http://isc-mirror.iris.washington.edu/cgi-bin/web-db-v4?request=COMPREHENSIVE&out_format=CATCSV&bot_lat=&top_lat=&left_lon=&right_lon=&searchshape=CIRC&ctr_lat=%.2f+&ctr_lon=%.2f&radius=40&max_dist_units=km&srn=&grn=&start_year=%d&start_month=%d&start_day=%d&start_time=%s&end_year=%d&end_month=%d&end_day=%d&end_time=%s&min_dep=&max_dep=&min_mag=&max_mag=&req_mag_type=Any&req_mag_agcy=Any&include_links=off"
    r = requests.get(url)

    page = r.content
    with open(os.path.join("scraped_events", "%d.txt" % ev.evid), 'w') as f:
        f.write(url+"\n")
        f.write(page)

    lon, lat, smaj, smin, strike, depth, depth_err = extract_ev(page)


    return lon, lat, smaj, smin, strike, depth, depth_err


from sigvisa.treegp.util import mkdir_p
mkdir_p("scraped_events")

s = load_events()

outfile = open("scraped.txt", 'w')
for i, (ev, (w, srate1)) in enumerate(s):
    try:
        lon, lat, smaj, smin, strike, depth, depth_err = scrape_isc(ev)
    except Exception as e:
        print e
        lon, lat, smaj, smin, strike, depth, depth_err = ev.lon, ev.lat, 20.0, 20.0, 0, ev.depth, 0.05*ev.depth + 1.0
    st = "%d, %d, %.4f, %.4f, %.1f, %.1f, %d, %.1f, %.1f" % (i, ev.evid, lon, lat, smaj, smin, strike, depth, depth_err)
    print st
    outfile.write(st + "\n")
    outfile.flush()
