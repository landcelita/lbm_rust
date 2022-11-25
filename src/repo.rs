use std::{collections::HashMap, fs::File};
use chrono::{DateTime, Utc};
use ndarray::{Array2, Array};
use ndarray_npy::ReadNpyExt;
use std::env;
use dotenv::dotenv;

#[derive(Hash, PartialEq, Eq)]
pub enum MeteorologicalType {
    UVert,
    UHori,
    Pressure,
}

fn get_meteorological_data(datetimes: Vec<DateTime<Utc>>) -> HashMap<(DateTime<Utc>, MeteorologicalType), Array2<f64>> {
    dotenv().ok();
    let data_dir = env::var("DATA_DIR").unwrap();
    let mut meteorological_data = HashMap::new();

    for datetime in datetimes {
        let u_vert_filename = data_dir.clone() + "npy/u_vert_" + &datetime.format("%Y%m%d%H").to_string() + ".npy";
        let u_hori_filename = data_dir.clone() + "npy/u_hori_" + &datetime.format("%Y%m%d%H").to_string() + ".npy";
        let pressure_filename = data_dir.clone() + "npy/pressure_" + &datetime.format("%Y%m%d%H").to_string() + ".npy";
        let reader = File::open(u_vert_filename).unwrap();
        let u_vert_arr = Array2::<f64>::read_npy(reader).unwrap();
        let reader = File::open(u_hori_filename).unwrap();
        let u_hori_arr = Array2::<f64>::read_npy(reader).unwrap();
        let reader = File::open(pressure_filename).unwrap();
        let pressure_arr = Array2::<f64>::read_npy(reader).unwrap();

        meteorological_data.insert((datetime, MeteorologicalType::UVert), u_vert_arr);
        meteorological_data.insert((datetime, MeteorologicalType::UHori), u_hori_arr);
        meteorological_data.insert((datetime, MeteorologicalType::Pressure), pressure_arr);
    }

    meteorological_data
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    #[test]
    fn test_get_meteorological_data() {
        let datetime1 = Utc.with_ymd_and_hms(2020, 3, 20, 3, 0, 0).unwrap();
        let datetime2 = Utc.with_ymd_and_hms(2018, 8, 29, 0, 0, 0).unwrap();
        let datetimes = vec![datetime1, datetime2];
        let meteorological_data = get_meteorological_data(datetimes);
        assert_eq!(
            -2.5597972869873047,
            *meteorological_data.get(&(datetime1, MeteorologicalType::UVert)).unwrap().get((3, 4)).unwrap()
        );
        assert_eq!(
            0.16127395629882812,
            *meteorological_data.get(&(datetime2, MeteorologicalType::UHori)).unwrap().get((10, 10)).unwrap()
        );
        assert_eq!(
            101150.23193359375,
            *meteorological_data.get(&(datetime2, MeteorologicalType::Pressure)).unwrap().get((6, 9)).unwrap()
        );
    }
}