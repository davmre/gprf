from datetime import datetime
import calendar
import requests
import os
import numpy as np


class CouldNotScrapeException(Exception):
    pass

import cPickle as pickle

TIMESTAMP_COL, TERR_COL, TRMS_COL, LON_COL, LAT_COL, SMAJ_COL, SMIN_COL, STRIKE_COL, DEPTH_COL, DERR_COL, METHOD_COL, SOURCE_COL, ISCID_COL, N_ISC_COLS = range(14)

def ev_from_line(line):

    try:
        evdate = line[:10]
        yr = int(evdate[:4])
        mo = int(evdate[5:7])
        day = int(evdate[8:])
        
        evtime = line[11:22]
        hr = int(evtime[:2])
        mn = int(evtime[3:5])
        ss = float(evtime[6:])
        s = int(ss)
        ms = float(ss - s) 

        dt = datetime(yr, mo, day, hr, mn, s)
        ts = calendar.timegm(dt.timetuple()) + ms
    except Exception as e:
        print "error parsing time", e
        ts = -1

    try:
        time_err = float(line[24:29])
    except:
        time_err = -1.0
    
    try:
        time_rms = float(line[30:35])
    except:
        time_rms = -1.0
    
    lat = float(line[36:44])
    lon = float(line[45:54])

    try:
        smaj = float(line[55:60])
        smin = float(line[61:66])
        strike = int(line[67:70])
    except:
        smaj = 20.0
        smin = 20.0
        strike = 0
    try:
        depth = float(line[71:76])
    except:
        depth = 0.0

    try:
        depth_err = float(line[78:82])
    except:
        depth_err = 0.05*depth + 1.0

    method = line[113]

    source = line[118:127].strip()

    try:
        iscid = int(line[129:136])
    except:
        iscid = -1

    return source, (ts, time_err, time_rms, lon, lat, smaj, smin, strike, depth, depth_err, method, source, iscid)

def extract_ev(page, target_ev=None):
    if "No events were found" in page:
        raise CouldNotScrapeException()

    try:
        idx1 = page.index("<pre>")+6
        idx2 = page.index("STOP")
        csvpage = page[idx1:idx2]
        lines = csvpage.split("\n")


        prime_hcenter = -1
        ev_hcenters = {}
        for line in lines:
            if "PRIME" in line:
                break

                if target_ev is not None and "IDC" in ev_hcenters and np.abs(ev_hcenters["IDC"][0] - target_ev.lon) > 1e-1:
                    # if the LEB lon for this event doesn't match the target, keep looking
                    ev_hcenters = {}                
                    continue
                else:
                    break
            if not line.startswith("20"): continue
            try:
                bulletin, hcenter = ev_from_line(line)
            except Exception as e:
                print "error parsing line", line, e
                import pdb; pdb.set_trace()
                continue
                
            ev_hcenters[bulletin] = hcenter

        if len(ev_hcenters)==0:
            raise CouldNotScrapeException()
        else:
            return ev_hcenters
    except Exception as e:
        print "error scraping", e
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

    hcenters = extract_ev(page, target_ev=ev)


    return hcenters

def fakescrape(ev):
    # for large datasets it becomes prohibitive to scrape the actual
    # ISC uncertainty for each event, so instead we return a rough
    # prior estimate based on the LEB location and the event
    # magnitude.

    # mb 3: 50km error
    # mb 4: 25km error
    # mb 5: 12km error
    # mb 6: 6km error
    # mb 2: 100km
    # mb 1: 200km
    # mb 0: 400km
    error_km = 400.0/(np.exp(ev.mb*np.log(2)))
    return ev.lon, ev.lat, error_km, error_km, 0, ev.depth, error_km

def oldmain():
    from gprf.seismic.seismic_util import load_events
    from sigvisa.treegp.util import mkdir_p
    mkdir_p("scraped_events")

    s = load_events(basedir="/home/dmoore/mkar_stuff")

    outfile = open("fakescraped.txt", 'w')
    for i, (ev, (w, srate1)) in enumerate(s):
        try:
            #lon, lat, smaj, smin, strike, depth, depth_err = scrape_isc(ev)
            lon, lat, smaj, smin, strike, depth, depth_err = fakescrape(ev)
        except Exception as e:
            print e
            lon, lat, smaj, smin, strike, depth, depth_err = ev.lon, ev.lat, 20.0, 20.0, 0, ev.depth, 0.05*ev.depth + 1.0
        st = "%d, %d, %.4f, %.4f, %.1f, %.1f, %d, %.1f, %.1f" % (i, ev.evid, lon, lat, smaj, smin, strike, depth, depth_err)
        print st
        outfile.write(st + "\n")
        outfile.flush()


def main():
    from gprf.seismic.seismic_util import load_events
    from sigvisa.treegp.util import mkdir_p



    def scrape_ev(i, ev, outfile_errs, outfile_isc, outfile_idc):
        try:
            hcenter_dict = scrape_isc(ev)
        except Exception as e:
            print "error scraping event", ev.evid, e
            outfile_errs.write("%d %s\n" % (ev.evid, e))
            outfile_errs.flush()
            return

        try:
            (ts, time_err, time_rms, lon, lat, smaj, smin, strike, depth, depth_err, method, source, iscid) = hcenter_dict['ISC']

            st = "%d, %d, %.2f, %.3f, %.4f, %.4f, %.1f, %.1f, %d, %.1f, %.1f" % (i, ev.evid, ts, time_err, lon, lat, smaj, smin, strike, depth, depth_err)
            print source, st
            outfile_isc.write(st + "\n")
            outfile_isc.flush()
        except Exception as e:
            print "ISC error", e


        try:
            (ts, time_err, time_rms, lon, lat, smaj, smin, strike, depth, depth_err, method, source, iscid) = hcenter_dict['IDC']

            st = "%d, %d, %.2f, %.3f, %.4f, %.4f, %.1f, %.1f, %d, %.1f, %.1f" % (i, ev.evid, ts, time_err, lon, lat, smaj, smin, strike, depth, depth_err)
            print source, st
            outfile_idc.write(st + "\n")
            outfile_idc.flush()
        except Exception as e:
            print "IDC error", e


        try:
            with open("scraped_events/full/%d.txt" % ev.evid, 'w') as f:
                f.write(repr(hcenter_dict))
        except Exception as e:
            print "repr error", e

    mkdir_p("scraped_events")
    mkdir_p("scraped_events/full/")
    outfile_errs = open("scraped_events/scrape_errors.txt", 'a')
    outfile_isc = open("scraped_events/isc.txt", 'a')
    outfile_idc = open("scraped_events/idc.txt", 'a')

    basedir="/home/dmoore/mkar_stuff"
    sta="mkar"
    start_bin = 1
    end_bin = 204
    for i in range(start_bin, end_bin):
        fname = os.path.join(basedir, "%s_stuff_%d" % (sta, i * 1000))
        with open(fname, 'rb') as f:
            ss = pickle.load(f)
            print "loaded", fname
            for i, (ev, (w, srate1)) in enumerate(ss):
                scrape_ev(i, ev, outfile_errs, outfile_isc, outfile_idc)

    outfile_idc.close()
    outfile_isc.close()
    outfile_errs.close()


if __name__=="__main__":
    main()
