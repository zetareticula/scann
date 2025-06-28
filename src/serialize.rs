// Copyright 2025 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Serialization utilities for converting between integers, floats, and binary keys.

use super::ScannError;
use std::error::Error;

fn uint_from_ieee754<FloatType, UintType>(f: FloatType) -> UintType
where
    FloatType: Into<u32> + Copy,
    UintType: From<u32> + std::ops::Not<Output = UintType> + std::ops::BitAnd<Output = UintType>,
{
    let n: u32 = unsafe { std::mem::transmute(f) };
    let sign_bit = !(!0u32 >> 1);
    if (n & sign_bit) == 0 {
        UintType::from(n + sign_bit)
    } else {
        UintType::from(0 - n)
    }
}

fn ieee754_from_uint<FloatType, UintType>(n: UintType) -> FloatType
where
    UintType: Into<u32> + std::ops::Not<Output = UintType> + std::ops::BitAnd<Output = UintType>,
    FloatType: From<u32>,
{
    let n: u32 = n.into();
    let sign_bit = !(!0u32 >> 1);
    let adjusted = if n & sign_bit != 0 {
        n - sign_bit
    } else {
        0 - n
    };
    unsafe { std::mem::transmute(adjusted) }
}

fn key_from_uint32(u32: u32, key: &mut Vec<u8>) {
    key.clear();
    key.extend_from_slice(&u32.to_be_bytes());
}

fn key_from_uint64(u64: u64, key: &mut Vec<u8>) {
    key.clear();
    key.extend_from_slice(&u64.to_be_bytes());
}

#[inline]
pub fn uint32_to_key(u32: u32) -> Vec<u8> {
    let mut key = Vec::new();
    key_from_uint32(u32, &mut key);
    key
}

#[inline]
pub fn int32_to_key(i32: i32) -> Vec<u8> {
    uint32_to_key(i32 as u32)
}

#[inline]
pub fn uint64_to_key(u64: u64) -> Vec<u8> {
    let mut key = Vec::new();
    key_from_uint64(u64, &mut key);
    key
}

pub fn key_to_uint32(key: &[u8]) -> Result<u32, Box<dyn Error>> {
    if key.len() != std::mem::size_of::<u32>() {
        return Err(Box::new(ScannError {
            message: format!("Invalid key length: expected {}, got {}", std::mem::size_of::<u32>(), key.len()),
        }));
    }
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(key);
    Ok(u32::from_be_bytes(bytes))
}

#[inline]
pub fn key_to_int32(key: &[u8]) -> Result<i32, Box<dyn Error>> {
    key_to_uint32(key).map(|v| v as i32)
}

pub fn key_to_uint64(key: &[u8]) -> Result<u64, Box<dyn Error>> {
    if key.len() != std::mem::size_of::<u64>() {
        return Err(Box::new(ScannError {
            message: format!("Invalid key length: expected {}, got {}", std::mem::size_of::<u64>(), key.len()),
        }));
    }
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(key);
    Ok(u64::from_be_bytes(bytes))
}

pub fn key_from_float(x: f32, key: &mut Vec<u8>) {
    let n = uint_from_ieee754::<f32, u32>(x);
    key_from_uint32(n, key);
}

#[inline]
pub fn float_to_key(x: f32) -> Vec<u8> {
    let mut key = Vec::new();
    key_from_float(x, &mut key);
    key
}

pub fn key_to_float(key: &[u8]) -> Result<f32, Box<dyn Error>> {
    let n = key_to_uint32(key)?;
    Ok(ieee754_from_uint::<f32, u32>(n))
}