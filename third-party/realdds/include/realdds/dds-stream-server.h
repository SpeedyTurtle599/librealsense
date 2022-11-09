// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2022 Intel Corporation. All Rights Reserved.

#pragma once

#include "dds-stream-profile.h"
#include "dds-stream-base.h"

#include <memory>
#include <string>


namespace realdds {


class dds_topic;
class dds_topic_writer;
class dds_publisher;
class dds_stream_profile;


struct image_header
{
    int format;
    int height = -1;
    int width = -1;

    bool is_valid() const { return width != -1 && height != -1; }
};


// Distributes stream images into a dedicated topic
// 
// This is a base class: you need to specify the type of stream via the instantiation of a video_stream_server, etc.
//
class dds_stream_server : public dds_stream_base
{
protected:
    dds_stream_server( std::string const & stream_name, std::string const & sensor_name );

public:
    virtual ~dds_stream_server();

    bool is_open() const override { return !! _writer; }
    virtual void open( std::string const & topic_name, std::shared_ptr< dds_publisher > const & ) = 0;

    bool is_streaming() const override { return _image_header.is_valid(); }
    void start_streaming( const image_header & header );
    void stop_streaming();

    void publish_image( const uint8_t * data, size_t size );

    std::shared_ptr< dds_topic > const & get_topic() const override;

protected:
    std::shared_ptr< dds_topic_writer > _writer;
    image_header _image_header;
};


class dds_video_stream_server : public dds_stream_server
{
    typedef dds_stream_server super;

public:
    dds_video_stream_server( std::string const & stream_name, std::string const & sensor_name );

    void open( std::string const & topic_name, std::shared_ptr< dds_publisher > const & ) override;

private:
    void check_profile( std::shared_ptr< dds_stream_profile > const & ) const override;
};


class dds_depth_stream_server : public dds_video_stream_server
{
    typedef dds_video_stream_server super;

public:
    dds_depth_stream_server( std::string const & stream_name, std::string const & sensor_name );

    char const * type_string() const override { return "depth"; }
};


class dds_ir_stream_server : public dds_video_stream_server
{
    typedef dds_video_stream_server super;

public:
    dds_ir_stream_server( std::string const & stream_name, std::string const & sensor_name );

    char const * type_string() const override { return "ir"; }
};


class dds_color_stream_server : public dds_video_stream_server
{
    typedef dds_video_stream_server super;

public:
    dds_color_stream_server( std::string const & stream_name, std::string const & sensor_name );

    char const * type_string() const override { return "color"; }
};


class dds_fisheye_stream_server : public dds_video_stream_server
{
    typedef dds_video_stream_server super;

public:
    dds_fisheye_stream_server( std::string const & stream_name, std::string const & sensor_name );

    char const * type_string() const override { return "fisheye"; }
};


class dds_confidence_stream_server : public dds_video_stream_server
{
    typedef dds_video_stream_server super;

public:
    dds_confidence_stream_server( std::string const & stream_name, std::string const & sensor_name );

    char const * type_string() const override { return "confidence"; }
};


class dds_motion_stream_server : public dds_stream_server
{
    typedef dds_stream_server super;

public:
    dds_motion_stream_server( std::string const & stream_name, std::string const & sensor_name );
    
    void open( std::string const & topic_name, std::shared_ptr< dds_publisher > const & ) override;

private:
    void check_profile( std::shared_ptr< dds_stream_profile > const & ) const override;
};


class dds_accel_stream_server : public dds_motion_stream_server
{
    typedef dds_motion_stream_server super;

public:
    dds_accel_stream_server( std::string const & stream_name, std::string const & sensor_name );

    char const * type_string() const override { return "accel"; }
};


class dds_gyro_stream_server : public dds_motion_stream_server
{
    typedef dds_motion_stream_server super;

public:
    dds_gyro_stream_server( std::string const & stream_name, std::string const & sensor_name );

    char const * type_string() const override { return "gyro"; }
};


class dds_pose_stream_server : public dds_motion_stream_server
{
    typedef dds_motion_stream_server super;

public:
    dds_pose_stream_server( std::string const & stream_name, std::string const & sensor_name );

    char const * type_string() const override { return "pose"; }
};


}  // namespace realdds
